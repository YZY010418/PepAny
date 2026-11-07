#!/usr/bin/env python3
"""Evaluate geometry adherence and cap-merge feasibility for PepAny raw attempts."""

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


RESULT_FIELDS = (
    'run_name',
    'target_id',
    'sample_index',
    'peptide_length',
    'length_error',
    'status',
    'evaluable',
    'ca_pass',
    'rmsd_pass',
    'both_pass',
    'ref_ca_distance',
    'gen_ca_distance',
    'ca_distance_deviation',
    'ca_distance_abs_deviation',
    'terminal_10atom_rmsd',
    'n_terminal_5atom_rmsd',
    'c_terminal_5atom_rmsd',
    'library_requested',
    'library_evaluable',
    'library_ca_pass',
    'library_rmsd_pass',
    'library_both_pass',
    'library_conformer_count',
    'library_invalid_conformer_count',
    'min_library_ca_abs_deviation',
    'best_library_ca_conformer',
    'min_library_10atom_rmsd',
    'best_library_rmsd_conformer',
    'best_library_joint_conformer',
    'best_library_joint_ca_abs_deviation',
    'best_library_joint_10atom_rmsd',
    'library_error',
    'library_align_no_clash_evaluable',
    'library_align_no_clash_pass',
    'library_rmsd_passing_conformer_count',
    'library_clash_checked_conformer_count',
    'library_clashing_conformer_count',
    'library_merge_error_count',
    'best_library_align_no_clash_conformer',
    'best_library_align_no_clash_10atom_rmsd',
    'library_align_no_clash_error',
    'ca_cutoff',
    'rmsd_cutoff',
    'clash_threshold',
    'metadata_file',
    'candidate_pdb',
    'candidate_sdf',
    'conditioning_conformer',
    'error',
    'copied_to',
    'copied_library_conformer',
    'copy_note',
)

COPY_CRITERION_FIELDS = {
    'ca': 'ca_pass',
    '10atom': 'rmsd_pass',
    'both': 'both_pass',
    'library-ca': 'library_ca_pass',
    'library-10atom': 'library_rmsd_pass',
    'library-both': 'library_both_pass',
    'library-align-no-clash': 'library_align_no_clash_pass',
}

COPY_LIBRARY_CONFORMER_FIELDS = {
    'library-ca': 'best_library_ca_conformer',
    'library-10atom': 'best_library_rmsd_conformer',
    'library-both': 'best_library_joint_conformer',
    'library-align-no-clash': 'best_library_align_no_clash_conformer',
}

VDW_RADII = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'P': 1.80,
    'S': 1.80,
    'F': 1.47,
    'CL': 1.75,
    'BR': 1.85,
    'I': 1.98,
}

WORKER_LIBRARY_PATHS = {}
WORKER_LIBRARY_CACHE = {}
WORKER_TARGET_RESIDUE_CACHE = {}


def load_sdf_mol(path, mol_index=0):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    if supplier is None:
        raise ValueError(f'Could not open SDF: {path}')
    molecules = [molecule for molecule in supplier if molecule is not None]
    if not molecules:
        raise ValueError(f'No valid molecule found in SDF: {path}')
    if mol_index < 0 or mol_index >= len(molecules):
        raise ValueError(
            f'Molecule index {mol_index} is out of range for '
            f'{len(molecules)} molecule(s): {path}'
        )
    molecule = molecules[mol_index]
    if molecule.GetNumConformers() == 0:
        raise ValueError(f'Molecule has no 3D conformer: {path}')
    return molecule


def get_positions(molecule, indexes):
    num_atoms = molecule.GetNumAtoms()
    invalid = [index for index in indexes if index < 0 or index >= num_atoms]
    if invalid:
        raise ValueError(
            f'Atom indexes out of range for molecule with {num_atoms} atoms: {invalid}'
        )
    conformer = molecule.GetConformer()
    coordinates = []
    for index in indexes:
        position = conformer.GetAtomPosition(int(index))
        coordinates.append([position.x, position.y, position.z])
    return np.asarray(coordinates, dtype=np.float64)


def atom_symbol(molecule, index):
    return molecule.GetAtomWithIdx(int(index)).GetSymbol()


def neighboring_atoms(molecule, index, symbol=None, exclude=None):
    excluded = set() if exclude is None else set(exclude)
    neighbors = []
    for neighbor in molecule.GetAtomWithIdx(int(index)).GetNeighbors():
        neighbor_index = neighbor.GetIdx()
        if neighbor_index in excluded:
            continue
        if symbol is not None and neighbor.GetSymbol() != symbol:
            continue
        neighbors.append(neighbor_index)
    return neighbors


def require_one(candidates, label):
    if len(candidates) != 1:
        raise ValueError(f'Could not uniquely infer {label}: candidates={candidates}')
    return candidates[0]


def infer_generated_indexes(molecule):
    """Infer the two terminal five-atom groups in a capped PepAny SDF."""
    num_atoms = molecule.GetNumAtoms()
    if num_atoms < 10:
        raise ValueError(
            f'Need at least 10 atoms to infer capped peptide termini, got {num_atoms}'
        )

    # PepAny writes ACE first as [CA, C, O] and NME last as [N, CA].
    n_terminal = [0, 1, 2, 3, 4]
    nme_n = num_atoms - 2
    nme_ca = num_atoms - 1
    if atom_symbol(molecule, nme_n) != 'N' or atom_symbol(molecule, nme_ca) != 'C':
        raise ValueError(
            'Could not infer NME atoms from the last two SDF atoms. The raw metadata '
            'must provide control.gen_atom_indices for a nonstandard atom order.'
        )

    last_c = require_one(
        neighboring_atoms(molecule, nme_n, symbol='C', exclude={nme_ca}),
        'last-residue carbonyl C bonded to NME.N',
    )
    oxygen_candidates = neighboring_atoms(
        molecule, last_c, symbol='O', exclude={nme_n}
    )
    double_bonded_oxygens = [
        index
        for index in oxygen_candidates
        if molecule.GetBondBetweenAtoms(last_c, index).GetBondType()
        == Chem.rdchem.BondType.DOUBLE
    ]
    last_o = require_one(
        double_bonded_oxygens or oxygen_candidates,
        'last-residue carbonyl O',
    )

    ca_candidates = neighboring_atoms(
        molecule,
        last_c,
        symbol='C',
        exclude={nme_n, last_o},
    )
    if last_c - 1 in ca_candidates:
        last_ca = last_c - 1
    else:
        last_ca = require_one(ca_candidates, 'last-residue CA')

    c_terminal = [last_ca, last_c, last_o, nme_n, nme_ca]
    return n_terminal + c_terminal


def kabsch_transform(moving, target):
    moving_mean = moving.mean(axis=0)
    target_mean = target.mean(axis=0)
    moving_centered = moving - moving_mean
    target_centered = target - target_mean
    covariance = moving_centered.T @ target_centered
    left_vectors, _, right_vectors = np.linalg.svd(covariance)
    if np.linalg.det(left_vectors) * np.linalg.det(right_vectors) < 0:
        left_vectors[:, -1] *= -1
    rotation = left_vectors @ right_vectors
    return rotation, moving_mean, target_mean


def kabsch_align(moving, target):
    rotation, moving_mean, target_mean = kabsch_transform(moving, target)
    return (moving - moving_mean) @ rotation + target_mean


def rmsd(left, right):
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=-1))))


def validate_slot_pair(slots, label):
    slots = tuple(int(slot) for slot in slots)
    if len(slots) != 2:
        raise ValueError(f'{label} must contain exactly two slots')
    invalid = [slot for slot in slots if slot < 0 or slot >= 10]
    if invalid:
        raise ValueError(f'{label} slots must be in [0, 9], got {invalid}')
    if slots[0] == slots[1]:
        raise ValueError(f'{label} slots must refer to different atoms')
    return slots


def pair_distance(coordinates, slots):
    left, right = slots
    return float(np.linalg.norm(coordinates[left] - coordinates[right]))


def all_positions(molecule):
    conformer = molecule.GetConformer()
    return np.asarray([
        list(conformer.GetAtomPosition(index))
        for index in range(molecule.GetNumAtoms())
    ], dtype=np.float64)


def set_atom_position(molecule, index, coordinate):
    molecule.GetConformer().SetAtomPosition(
        int(index),
        Point3D(*(float(value) for value in coordinate)),
    )


def remove_atoms_with_map(molecule, remove_indexes):
    remove_indexes = {int(index) for index in remove_indexes}
    invalid = [
        index for index in remove_indexes
        if index < 0 or index >= molecule.GetNumAtoms()
    ]
    if invalid:
        raise ValueError(f'Cannot remove atom indexes out of range: {invalid}')

    old_to_new = {}
    for index in range(molecule.GetNumAtoms()):
        if index not in remove_indexes:
            old_to_new[index] = len(old_to_new)

    editable = Chem.RWMol(molecule)
    for index in sorted(remove_indexes, reverse=True):
        editable.RemoveAtom(index)
    return editable.GetMol(), old_to_new


def heavy_atom_mol_with_map(molecule):
    hydrogens = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol().upper() in {'H', 'D'}
    ]
    if not hydrogens:
        return Chem.Mol(molecule), {
            index: index for index in range(molecule.GetNumAtoms())
        }
    return remove_atoms_with_map(molecule, hydrogens)


def remap_indexes(indexes, old_to_new, label):
    remapped = []
    for index in indexes:
        index = int(index)
        if index not in old_to_new:
            raise ValueError(
                f'{label} atom index {index} is hydrogen/deuterium and cannot '
                'be used for terminal merge'
            )
        remapped.append(old_to_new[index])
    return remapped


def align_reference_to_generated(ref_mol, ref_indexes, gen_mol, gen_indexes):
    ref_work = Chem.Mol(ref_mol)
    gen_work = Chem.Mol(gen_mol)
    rotation, ref_mean, gen_mean = kabsch_transform(
        get_positions(ref_work, ref_indexes),
        get_positions(gen_work, gen_indexes),
    )
    aligned_ref_coords = (all_positions(ref_work) - ref_mean) @ rotation + gen_mean
    for index, coordinate in enumerate(aligned_ref_coords):
        set_atom_position(ref_work, index, coordinate)

    # Match CPSea2: average fitted terminal anchors before making the new bonds.
    fitted_ref_anchors = get_positions(ref_work, ref_indexes)
    for gen_index, ref_index, ref_coordinate in zip(
        gen_indexes, ref_indexes, fitted_ref_anchors
    ):
        gen_coordinate = get_positions(gen_work, [gen_index])[0]
        averaged = (gen_coordinate + ref_coordinate) * 0.5
        set_atom_position(gen_work, gen_index, averaged)
        set_atom_position(ref_work, ref_index, averaged)
    return ref_work, gen_work


def merge_terminal_cap(ref_mol, ref_indexes, gen_mol, gen_indexes):
    """Replace generated ACE/NME placeholders with one aligned CPcap conformer."""
    ref_mol, ref_old_to_new = heavy_atom_mol_with_map(ref_mol)
    gen_mol, gen_old_to_new = heavy_atom_mol_with_map(gen_mol)
    ref_indexes = remap_indexes(ref_indexes, ref_old_to_new, 'Reference')
    gen_indexes = remap_indexes(gen_indexes, gen_old_to_new, 'Generated')

    ref_n = list(ref_indexes[:5])
    ref_c = list(ref_indexes[5:])
    gen_n = list(gen_indexes[:5])
    gen_c = list(gen_indexes[5:])
    ref_work, gen_work = align_reference_to_generated(
        ref_mol, ref_indexes, gen_mol, gen_indexes
    )

    gen_kept, gen_map = remove_atoms_with_map(
        gen_work, set(gen_n[:3] + gen_c[3:])
    )
    ref_kept, ref_map = remove_atoms_with_map(
        ref_work, set(ref_n[3:] + ref_c[:3])
    )
    ref_offset = gen_kept.GetNumAtoms()
    merged = Chem.RWMol(Chem.CombineMols(gen_kept, ref_kept))
    new_bonds = (
        (gen_n[3], ref_n[1]),
        (gen_c[1], ref_c[3]),
    )
    for gen_index, ref_index in new_bonds:
        if gen_index not in gen_map or ref_index not in ref_map:
            raise ValueError('A terminal link atom was removed unexpectedly')
        merged.AddBond(
            int(gen_map[gen_index]),
            int(ref_offset + ref_map[ref_index]),
            Chem.rdchem.BondType.SINGLE,
        )

    merged = merged.GetMol()
    try:
        Chem.SanitizeMol(merged)
    except Exception:
        pass
    return merged


def vdw_radius(symbol):
    return VDW_RADII.get(str(symbol).upper(), 1.80)


def detect_heavy_atom_clashes(molecule, threshold):
    """Return nonlocal heavy-atom pairs closer than scaled VDW radii."""
    conformer = molecule.GetConformer()
    heavy_indexes = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol().upper() not in {'H', 'D'}
    ]
    topological_distances = Chem.GetDistanceMatrix(molecule)
    clashes = []
    for position, left_index in enumerate(heavy_indexes):
        left_atom = molecule.GetAtomWithIdx(left_index)
        left_coordinate = np.asarray(
            conformer.GetAtomPosition(left_index), dtype=np.float64
        )
        for right_index in heavy_indexes[position + 1:]:
            if 0 < topological_distances[left_index, right_index] <= 3:
                continue
            right_atom = molecule.GetAtomWithIdx(right_index)
            right_coordinate = np.asarray(
                conformer.GetAtomPosition(right_index), dtype=np.float64
            )
            distance = float(np.linalg.norm(left_coordinate - right_coordinate))
            cutoff = float(threshold) * (
                vdw_radius(left_atom.GetSymbol())
                + vdw_radius(right_atom.GetSymbol())
            )
            if distance <= cutoff:
                clashes.append((left_index, right_index, distance, cutoff))
    return clashes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Calculate C-alpha distance and terminal 10-atom RMSD success rates for '
            'filter-free PepAny raw outputs. Every raw metadata JSON is retained in '
            'the denominator, including attempts whose structural serialization failed.'
        )
    )
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help=(
            'PepAny raw output root, one cyclization-type directory, or one target '
            'directory. raw_model_outputs/*.json files are discovered recursively.'
        ),
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        default=None,
        help='Report directory. Default: <input>/geometry_evaluation',
    )
    parser.add_argument(
        '--ca-cutoff',
        type=float,
        default=1.0,
        help=(
            'Pass when the absolute C-alpha distance error is below this value '
            'in A. Default: 1.0'
        ),
    )
    parser.add_argument(
        '--rmsd-cutoff',
        type=float,
        default=1.0,
        help=(
            'Pass when the Kabsch-aligned terminal 10-atom RMSD is below this '
            'value in A. Default: 1.0'
        ),
    )
    parser.add_argument(
        '--candidate-mol-index',
        type=int,
        default=0,
        help='Molecule index in each generated candidate SDF. Default: 0',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Evaluation worker processes; use 0 for all available CPUs. Default: 1',
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=8,
        help='Attempts assigned to each process-pool task batch. Default: 8',
    )
    parser.add_argument(
        '--expected-attempts-per-target',
        type=int,
        default=None,
        help='Optional transfer-completeness check, for example 100.',
    )
    parser.add_argument(
        '--grouped-success-rates',
        action='store_true',
        help=(
            'Also report success and failure-mode rates by cyclization type, '
            'peptide length, and cyclization-type/length combination.'
        ),
    )
    parser.add_argument(
        '--capping-library-root',
        default=None,
        help=(
            'Optional CPcap root containing one subdirectory per run/cyclization '
            'name. All matching SDFs under <root>/<run_name> are compared.'
        ),
    )
    parser.add_argument(
        '--capping-library',
        action='append',
        default=[],
        metavar='RUN_NAME=PATH',
        help=(
            'Override or define a capping library for one run. PATH may be an SDF '
            'file or directory. Repeat for multiple run names.'
        ),
    )
    parser.add_argument(
        '--library-pattern',
        default='*.sdf',
        help='Recursive filename pattern inside each capping-library directory. Default: *.sdf',
    )
    parser.add_argument(
        '--clash-threshold',
        type=float,
        default=0.8,
        help=(
            'Heavy-atom clash cutoff as a scale on the sum of VDW radii after '
            'merging an RMSD-passing CPcap conformer. Default: 0.8'
        ),
    )
    parser.add_argument(
        '--copy-passed-to',
        default=None,
        help=(
            'Optional new directory for attempts passing --copy-criterion. By default '
            'no structures are copied.'
        ),
    )
    parser.add_argument(
        '--copy-criterion',
        choices=tuple(COPY_CRITERION_FIELDS),
        default='both',
        help=(
            'Criterion used by --copy-passed-to: ca for CA-distance success, '
            '10atom for terminal 10-atom RMSD success, both for their intersection, '
            'or a library-* any-conformer criterion including '
            'library-align-no-clash. Default: both'
        ),
    )
    parser.add_argument(
        '--include-raw-tensors',
        action='store_true',
        help='Also copy raw_model_outputs/<sample>.pt when --copy-passed-to is used.',
    )
    return parser.parse_args()


def validate_args(args):
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {input_dir}')
    if args.ca_cutoff <= 0:
        raise ValueError('--ca-cutoff must be positive')
    if args.rmsd_cutoff <= 0:
        raise ValueError('--rmsd-cutoff must be positive')
    if args.candidate_mol_index < 0:
        raise ValueError('--candidate-mol-index must be non-negative')
    if args.workers < 0:
        raise ValueError('--workers must be >= 0')
    if args.chunksize <= 0:
        raise ValueError('--chunksize must be positive')
    if (
        args.expected_attempts_per_target is not None
        and args.expected_attempts_per_target <= 0
    ):
        raise ValueError('--expected-attempts-per-target must be positive')
    if not args.library_pattern:
        raise ValueError('--library-pattern cannot be empty')
    if args.clash_threshold <= 0:
        raise ValueError('--clash-threshold must be positive')
    library_requested = bool(args.capping_library_root or args.capping_library)
    if args.copy_criterion.startswith('library-') and not library_requested:
        raise ValueError(
            f'--copy-criterion {args.copy_criterion} requires '
            '--capping-library-root or --capping-library'
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else input_dir / 'geometry_evaluation'
    )
    copy_dir = Path(args.copy_passed_to).resolve() if args.copy_passed_to else None
    if copy_dir is not None:
        if copy_dir == input_dir or input_dir in copy_dir.parents:
            raise ValueError('--copy-passed-to must be outside the raw input directory')
        if copy_dir.exists() and any(copy_dir.iterdir()):
            raise FileExistsError(
                f'Copy destination is not empty: {copy_dir}. Use a new directory to '
                'avoid mixing subsets from different evaluations.'
            )
    return input_dir, output_dir, copy_dir


def parse_library_overrides(specifications):
    overrides = {}
    for specification in specifications:
        if '=' not in specification:
            raise ValueError(
                f'Invalid --capping-library {specification!r}; expected RUN_NAME=PATH'
            )
        run_name, raw_path = specification.split('=', 1)
        run_name = run_name.strip()
        raw_path = raw_path.strip()
        if not run_name or not raw_path:
            raise ValueError(
                f'Invalid --capping-library {specification!r}; expected RUN_NAME=PATH'
            )
        if run_name in overrides:
            raise ValueError(f'Duplicate --capping-library definition for {run_name!r}')
        overrides[run_name] = Path(raw_path).resolve()
    return overrides


def collect_library_sdfs(path, pattern):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Capping library path does not exist: {path}')
    return sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file())


def resolve_library_paths(jobs, args):
    overrides = parse_library_overrides(args.capping_library)
    root = Path(args.capping_library_root).resolve() if args.capping_library_root else None
    if root is None and not overrides:
        return {}, {}
    if root is not None and not root.is_dir():
        raise NotADirectoryError(f'Capping library root does not exist: {root}')

    run_names = sorted({job['run_name'] for job in jobs})
    paths_by_run = {}
    sources_by_run = {}
    missing = []
    for run_name in run_names:
        if run_name in overrides:
            source = overrides[run_name]
        elif root is not None:
            source = root / run_name
            if not source.exists() and len(run_names) == 1:
                direct_sdfs = sorted(
                    candidate for candidate in root.glob(args.library_pattern)
                    if candidate.is_file()
                )
                if direct_sdfs:
                    source = root
        else:
            missing.append(run_name)
            continue

        try:
            sdf_paths = collect_library_sdfs(source, args.library_pattern)
        except FileNotFoundError:
            missing.append(run_name)
            continue
        if not sdf_paths:
            raise FileNotFoundError(
                f'No SDF files matching {args.library_pattern!r} found for '
                f'{run_name!r} under: {source}'
            )
        paths_by_run[run_name] = [str(path) for path in sdf_paths]
        sources_by_run[run_name] = str(source)

    if missing:
        formatted = ', '.join(repr(run_name) for run_name in missing)
        raise FileNotFoundError(
            f'No capping library configured for run(s): {formatted}. Expected '
            '<capping-library-root>/<run_name> or an explicit '
            '--capping-library RUN_NAME=PATH mapping.'
        )
    return paths_by_run, sources_by_run


def zero_based_indexes(indexes, index_base, label):
    if indexes is None or len(indexes) != 10:
        raise ValueError(f'{label} must contain exactly 10 atom indexes')
    return [int(index) - int(index_base) for index in indexes]


def local_artifact_path(target_dir, metadata, key, fallback):
    recorded = metadata.get(key)
    filename = Path(recorded).name if recorded else fallback
    return target_dir / filename


def conditioning_conformer_path(target_dir, metadata, sample_stem):
    recorded = metadata.get('conditioning_conformer')
    filename = Path(recorded).name if recorded else f'{sample_stem}.sdf'
    return target_dir / 'conditioning_conformers' / filename


def pdb_residue_key(line):
    return (
        line[21:22].strip(),
        line[22:26].strip(),
        line[26:27].strip(),
    )


def pdb_residue_keys(path, atom_records_only=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing PDB file: {path}')
    residue_keys = set()
    with path.open() as handle:
        for line in handle:
            if atom_records_only:
                if not line.startswith('ATOM  '):
                    continue
            elif not line.startswith(('ATOM  ', 'HETATM')):
                continue
            residue_keys.add(pdb_residue_key(line))
    return residue_keys


def count_generated_peptide_residues(candidate_pdb, target_pdb):
    target_cache_key = str(Path(target_pdb).resolve())
    if target_cache_key not in WORKER_TARGET_RESIDUE_CACHE:
        WORKER_TARGET_RESIDUE_CACHE[target_cache_key] = pdb_residue_keys(target_pdb)
    target_residues = WORKER_TARGET_RESIDUE_CACHE[target_cache_key]
    candidate_amino_acids = pdb_residue_keys(candidate_pdb, atom_records_only=True)
    generated_residues = candidate_amino_acids - target_residues
    if not generated_residues:
        raise ValueError(
            'Could not identify generated standard-amino-acid residues by subtracting '
            'target.pdb residues from the candidate PDB'
        )
    return len(generated_residues)


def load_job(metadata_path, input_dir, args):
    metadata_path = Path(metadata_path)
    target_dir = metadata_path.parent.parent
    relative_target_dir = target_dir.relative_to(input_dir)
    sample_stem = metadata_path.stem
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        metadata = {}
        metadata_error = f'Could not read metadata: {exc}'
    else:
        metadata_error = ''

    sample_index = metadata.get('sample_index', sample_stem)
    candidate_pdb = local_artifact_path(
        target_dir, metadata, 'candidate_pdb', f'{sample_stem}.pdb'
    )
    candidate_sdf = local_artifact_path(
        target_dir, metadata, 'candidate_sdf', f'{sample_stem}.sdf'
    )
    conformer = conditioning_conformer_path(target_dir, metadata, sample_stem)
    return {
        'run_name': str(metadata.get(
            'run_name',
            relative_target_dir.parts[0] if relative_target_dir.parts else '',
        )),
        'target_id': str(metadata.get('target_id', target_dir.name)),
        'sample_index': sample_index,
        'metadata_file': str(metadata_path),
        'target_dir': str(target_dir),
        'relative_target_dir': str(relative_target_dir),
        'candidate_pdb': str(candidate_pdb),
        'candidate_sdf': str(candidate_sdf),
        'conditioning_conformer': str(conformer),
        'control': metadata.get('control') or {},
        'metadata_error': metadata_error,
        'ca_cutoff': float(args.ca_cutoff),
        'rmsd_cutoff': float(args.rmsd_cutoff),
        'candidate_mol_index': int(args.candidate_mol_index),
        'grouped_success_rates': bool(args.grouped_success_rates),
        'clash_threshold': float(args.clash_threshold),
    }


def empty_result(job):
    return {
        'run_name': job['run_name'],
        'target_id': job['target_id'],
        'sample_index': job['sample_index'],
        'peptide_length': '',
        'length_error': '',
        'status': 'error',
        'evaluable': False,
        'ca_pass': False,
        'rmsd_pass': False,
        'both_pass': False,
        'ref_ca_distance': '',
        'gen_ca_distance': '',
        'ca_distance_deviation': '',
        'ca_distance_abs_deviation': '',
        'terminal_10atom_rmsd': '',
        'n_terminal_5atom_rmsd': '',
        'c_terminal_5atom_rmsd': '',
        'library_requested': bool(job.get('library_requested', False)),
        'library_evaluable': False,
        'library_ca_pass': False,
        'library_rmsd_pass': False,
        'library_both_pass': False,
        'library_conformer_count': 0,
        'library_invalid_conformer_count': 0,
        'min_library_ca_abs_deviation': '',
        'best_library_ca_conformer': '',
        'min_library_10atom_rmsd': '',
        'best_library_rmsd_conformer': '',
        'best_library_joint_conformer': '',
        'best_library_joint_ca_abs_deviation': '',
        'best_library_joint_10atom_rmsd': '',
        'library_error': '',
        'library_align_no_clash_evaluable': False,
        'library_align_no_clash_pass': False,
        'library_rmsd_passing_conformer_count': 0,
        'library_clash_checked_conformer_count': 0,
        'library_clashing_conformer_count': 0,
        'library_merge_error_count': 0,
        'best_library_align_no_clash_conformer': '',
        'best_library_align_no_clash_10atom_rmsd': '',
        'library_align_no_clash_error': '',
        'ca_cutoff': job['ca_cutoff'],
        'rmsd_cutoff': job['rmsd_cutoff'],
        'clash_threshold': job['clash_threshold'],
        'metadata_file': job['metadata_file'],
        'candidate_pdb': job['candidate_pdb'],
        'candidate_sdf': job['candidate_sdf'],
        'conditioning_conformer': job['conditioning_conformer'],
        'error': '',
        'copied_to': '',
        'copied_library_conformer': '',
        'copy_note': '',
    }


def init_evaluation_worker(library_paths_by_run):
    global WORKER_LIBRARY_PATHS
    global WORKER_LIBRARY_CACHE
    global WORKER_TARGET_RESIDUE_CACHE
    WORKER_LIBRARY_PATHS = {
        run_name: tuple(paths)
        for run_name, paths in library_paths_by_run.items()
    }
    WORKER_LIBRARY_CACHE = {}
    WORKER_TARGET_RESIDUE_CACHE = {}


def load_worker_library(job, ref_indexes, ref_slots):
    run_name = job['run_name']
    paths = WORKER_LIBRARY_PATHS.get(run_name)
    if paths is None:
        return None

    ref_mol_index = int(job['control'].get('term_rel_pos_mol_index', 0))
    cache_key = (run_name, tuple(ref_indexes), tuple(ref_slots), ref_mol_index)
    if cache_key in WORKER_LIBRARY_CACHE:
        return WORKER_LIBRARY_CACHE[cache_key]

    conformers = []
    invalid_count = 0
    for path in paths:
        try:
            molecule = load_sdf_mol(path, ref_mol_index)
            coordinates = get_positions(molecule, ref_indexes)
            conformers.append({
                'path': path,
                'coordinates': coordinates,
                'ca_distance': pair_distance(coordinates, ref_slots),
            })
        except Exception:
            invalid_count += 1

    if not conformers:
        raise ValueError(
            f'All {len(paths)} capping-library SDFs failed to load for {run_name!r}'
        )
    cached = {
        'conformers': conformers,
        'invalid_count': invalid_count,
    }
    WORKER_LIBRARY_CACHE[cache_key] = cached
    return cached


def evaluate_library_align_no_clash(
    job,
    gen_mol,
    gen_indexes,
    ref_indexes,
    conformers,
    rmsd_values,
):
    passing_indexes = np.flatnonzero(rmsd_values < job['rmsd_cutoff'])
    passing_indexes = sorted(passing_indexes, key=lambda index: rmsd_values[index])
    outcome = {
        'library_align_no_clash_evaluable': len(passing_indexes) == 0,
        'library_align_no_clash_pass': False,
        'library_rmsd_passing_conformer_count': len(passing_indexes),
        'library_clash_checked_conformer_count': 0,
        'library_clashing_conformer_count': 0,
        'library_merge_error_count': 0,
        'best_library_align_no_clash_conformer': '',
        'best_library_align_no_clash_10atom_rmsd': '',
        'library_align_no_clash_error': '',
    }
    merge_errors = []
    ref_mol_index = int(job['control'].get('term_rel_pos_mol_index', 0))
    for index in passing_indexes:
        conformer = conformers[index]
        try:
            ref_mol = load_sdf_mol(conformer['path'], ref_mol_index)
            merged = merge_terminal_cap(
                ref_mol,
                ref_indexes,
                gen_mol,
                gen_indexes,
            )
            clashes = detect_heavy_atom_clashes(
                merged, threshold=job['clash_threshold']
            )
        except Exception as exc:
            outcome['library_merge_error_count'] += 1
            merge_errors.append(f"{conformer['path']}: {exc}")
            continue

        outcome['library_clash_checked_conformer_count'] += 1
        if clashes:
            outcome['library_clashing_conformer_count'] += 1
            continue

        outcome.update({
            'library_align_no_clash_evaluable': True,
            'library_align_no_clash_pass': True,
            'best_library_align_no_clash_conformer': conformer['path'],
            'best_library_align_no_clash_10atom_rmsd': float(rmsd_values[index]),
        })
        break

    if not outcome['library_align_no_clash_pass'] and passing_indexes:
        outcome['library_align_no_clash_evaluable'] = (
            outcome['library_merge_error_count'] == 0
        )
    if merge_errors:
        preview = merge_errors[:3]
        if len(merge_errors) > len(preview):
            preview.append(f'... and {len(merge_errors) - len(preview)} more')
        outcome['library_align_no_clash_error'] = '; '.join(preview)
    return outcome


def evaluate_library_geometry(
    job,
    gen_mol,
    gen_indexes,
    gen_coords,
    gen_ca_distance,
    ref_indexes,
    ref_slots,
):
    library = load_worker_library(job, ref_indexes, ref_slots)
    if library is None:
        return {}

    conformers = library['conformers']
    ca_errors = np.asarray([
        abs(gen_ca_distance - conformer['ca_distance'])
        for conformer in conformers
    ], dtype=np.float64)
    rmsd_values = np.asarray([
        rmsd(kabsch_align(gen_coords, conformer['coordinates']), conformer['coordinates'])
        for conformer in conformers
    ], dtype=np.float64)

    ca_best_index = int(np.argmin(ca_errors))
    rmsd_best_index = int(np.argmin(rmsd_values))
    joint_scores = np.maximum(
        ca_errors / job['ca_cutoff'],
        rmsd_values / job['rmsd_cutoff'],
    )
    joint_best_index = int(np.argmin(joint_scores))
    ca_passes = ca_errors < job['ca_cutoff']
    rmsd_passes = rmsd_values < job['rmsd_cutoff']

    outcome = {
        'library_evaluable': True,
        'library_ca_pass': bool(np.any(ca_passes)),
        'library_rmsd_pass': bool(np.any(rmsd_passes)),
        'library_both_pass': bool(np.any(ca_passes & rmsd_passes)),
        'library_conformer_count': len(conformers),
        'library_invalid_conformer_count': library['invalid_count'],
        'min_library_ca_abs_deviation': float(ca_errors[ca_best_index]),
        'best_library_ca_conformer': conformers[ca_best_index]['path'],
        'min_library_10atom_rmsd': float(rmsd_values[rmsd_best_index]),
        'best_library_rmsd_conformer': conformers[rmsd_best_index]['path'],
        'best_library_joint_conformer': conformers[joint_best_index]['path'],
        'best_library_joint_ca_abs_deviation': float(ca_errors[joint_best_index]),
        'best_library_joint_10atom_rmsd': float(rmsd_values[joint_best_index]),
    }
    outcome.update(evaluate_library_align_no_clash(
        job,
        gen_mol,
        gen_indexes,
        ref_indexes,
        conformers,
        rmsd_values,
    ))
    return outcome


def evaluate_attempt(job):
    result = empty_result(job)
    if job['grouped_success_rates']:
        try:
            result['peptide_length'] = count_generated_peptide_residues(
                job['candidate_pdb'],
                Path(job['target_dir']) / 'target.pdb',
            )
        except Exception as exc:
            result['peptide_length'] = 'unknown'
            result['length_error'] = str(exc)

    try:
        if job['metadata_error']:
            raise ValueError(job['metadata_error'])

        control = job['control']
        ref_indexes = zero_based_indexes(
            control.get('term_rel_pos_atom_indices'),
            control.get('term_rel_pos_index_base', 0),
            'control.term_rel_pos_atom_indices',
        )
        ref_slots = validate_slot_pair(
            control.get('ref_ca_distance_slots', (0, 9)),
            'control.ref_ca_distance_slots',
        )
        gen_slots = validate_slot_pair(
            control.get('gen_ca_distance_slots', (0, 9)),
            'control.gen_ca_distance_slots',
        )

        candidate_path = Path(job['candidate_sdf'])
        if not candidate_path.is_file():
            raise FileNotFoundError(f'Missing candidate SDF: {candidate_path}')

        gen_mol = load_sdf_mol(candidate_path, job['candidate_mol_index'])
        gen_index_values = control.get('gen_atom_indices')
        if gen_index_values is None:
            gen_indexes = infer_generated_indexes(gen_mol)
        else:
            gen_indexes = zero_based_indexes(
                gen_index_values,
                control.get('gen_index_base', 0),
                'control.gen_atom_indices',
            )

        gen_coords = get_positions(gen_mol, gen_indexes)
        gen_ca_distance = pair_distance(gen_coords, gen_slots)
        result['gen_ca_distance'] = gen_ca_distance
    except Exception as exc:
        result['error'] = str(exc)
        return result

    try:
        conformer_path = Path(job['conditioning_conformer'])
        if not conformer_path.is_file():
            raise FileNotFoundError(f'Missing conditioning conformer: {conformer_path}')
        ref_mol = load_sdf_mol(
            conformer_path,
            int(control.get('term_rel_pos_mol_index', 0)),
        )
        ref_coords = get_positions(ref_mol, ref_indexes)
        aligned = kabsch_align(gen_coords, ref_coords)
        terminal_rmsd = rmsd(aligned, ref_coords)
        n_terminal_rmsd = rmsd(aligned[:5], ref_coords[:5])
        c_terminal_rmsd = rmsd(aligned[5:], ref_coords[5:])
        ref_ca_distance = pair_distance(ref_coords, ref_slots)
        ca_deviation = gen_ca_distance - ref_ca_distance
        ca_abs_deviation = abs(ca_deviation)
        ca_pass = ca_abs_deviation < job['ca_cutoff']
        rmsd_pass = terminal_rmsd < job['rmsd_cutoff']

        result.update({
            'status': 'pass' if ca_pass and rmsd_pass else 'fail',
            'evaluable': True,
            'ca_pass': ca_pass,
            'rmsd_pass': rmsd_pass,
            'both_pass': ca_pass and rmsd_pass,
            'ref_ca_distance': ref_ca_distance,
            'gen_ca_distance': gen_ca_distance,
            'ca_distance_deviation': ca_deviation,
            'ca_distance_abs_deviation': ca_abs_deviation,
            'terminal_10atom_rmsd': terminal_rmsd,
            'n_terminal_5atom_rmsd': n_terminal_rmsd,
            'c_terminal_5atom_rmsd': c_terminal_rmsd,
        })
    except Exception as exc:
        result['error'] = str(exc)

    try:
        result.update(evaluate_library_geometry(
            job,
            gen_mol,
            gen_indexes,
            gen_coords,
            gen_ca_distance,
            ref_indexes,
            ref_slots,
        ))
    except Exception as exc:
        result['library_error'] = str(exc)
    return result


def resolve_workers(requested, n_jobs):
    if n_jobs == 0:
        return 1
    available = os.cpu_count() or 1
    workers = available if requested == 0 else requested
    return max(1, min(int(workers), n_jobs))


def evaluate_jobs(jobs, workers, chunksize, library_paths_by_run):
    if workers == 1:
        init_evaluation_worker(library_paths_by_run)
        iterator = map(evaluate_attempt, jobs)
        return list(tqdm(iterator, total=len(jobs), desc='PepAny raw geometry'))

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_evaluation_worker,
        initargs=(library_paths_by_run,),
    ) as executor:
        iterator = executor.map(evaluate_attempt, jobs, chunksize=chunksize)
        return list(tqdm(iterator, total=len(jobs), desc='PepAny raw geometry'))


def rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def summarize_rows(rows):
    attempts = len(rows)
    evaluable = sum(bool(row['evaluable']) for row in rows)
    ca_successes = sum(bool(row['ca_pass']) for row in rows)
    rmsd_successes = sum(bool(row['rmsd_pass']) for row in rows)
    both_successes = sum(bool(row['both_pass']) for row in rows)
    library_requested = any(bool(row['library_requested']) for row in rows)
    library_evaluable = sum(bool(row['library_evaluable']) for row in rows)
    library_ca_successes = sum(bool(row['library_ca_pass']) for row in rows)
    library_rmsd_successes = sum(bool(row['library_rmsd_pass']) for row in rows)
    library_both_successes = sum(bool(row['library_both_pass']) for row in rows)
    library_align_no_clash_evaluable = sum(
        bool(row['library_align_no_clash_evaluable']) for row in rows
    )
    library_align_no_clash_successes = sum(
        bool(row['library_align_no_clash_pass']) for row in rows
    )
    condition_ca_only = sum(
        bool(row['evaluable'] and row['ca_pass'] and not row['rmsd_pass'])
        for row in rows
    )
    condition_rmsd_only = sum(
        bool(row['evaluable'] and row['rmsd_pass'] and not row['ca_pass'])
        for row in rows
    )
    condition_neither = sum(
        bool(row['evaluable'] and not row['ca_pass'] and not row['rmsd_pass'])
        for row in rows
    )
    library_split_only = sum(
        bool(
            row['library_evaluable']
            and row['library_ca_pass']
            and row['library_rmsd_pass']
            and not row['library_both_pass']
        )
        for row in rows
    )
    library_ca_only = sum(
        bool(
            row['library_evaluable']
            and row['library_ca_pass']
            and not row['library_rmsd_pass']
        )
        for row in rows
    )
    library_rmsd_only = sum(
        bool(
            row['library_evaluable']
            and row['library_rmsd_pass']
            and not row['library_ca_pass']
        )
        for row in rows
    )
    library_neither = sum(
        bool(
            row['library_evaluable']
            and not row['library_ca_pass']
            and not row['library_rmsd_pass']
        )
        for row in rows
    )
    return {
        'attempts': attempts,
        'evaluable': evaluable,
        'evaluation_errors': attempts - evaluable,
        'evaluable_rate': rate(evaluable, attempts),
        'ca_successes': ca_successes,
        'ca_success_rate_all_attempts': rate(ca_successes, attempts),
        'ca_success_rate_evaluable': rate(ca_successes, evaluable),
        'rmsd_successes': rmsd_successes,
        'rmsd_success_rate_all_attempts': rate(rmsd_successes, attempts),
        'rmsd_success_rate_evaluable': rate(rmsd_successes, evaluable),
        'both_successes': both_successes,
        'both_success_rate_all_attempts': rate(both_successes, attempts),
        'both_success_rate_evaluable': rate(both_successes, evaluable),
        'condition_ca_only_count': condition_ca_only,
        'condition_ca_only_rate_all_attempts': rate(condition_ca_only, attempts),
        'condition_ca_only_rate_evaluable': rate(condition_ca_only, evaluable),
        'condition_rmsd_only_count': condition_rmsd_only,
        'condition_rmsd_only_rate_all_attempts': rate(
            condition_rmsd_only, attempts
        ),
        'condition_rmsd_only_rate_evaluable': rate(condition_rmsd_only, evaluable),
        'condition_neither_count': condition_neither,
        'condition_neither_rate_all_attempts': rate(condition_neither, attempts),
        'condition_neither_rate_evaluable': rate(condition_neither, evaluable),
        'library_evaluable': library_evaluable,
        'library_evaluation_errors': (
            attempts - library_evaluable if library_requested else 0
        ),
        'library_evaluable_rate': rate(library_evaluable, attempts),
        'library_ca_successes': library_ca_successes,
        'library_ca_success_rate_all_attempts': rate(library_ca_successes, attempts),
        'library_ca_success_rate_evaluable': rate(
            library_ca_successes, library_evaluable
        ),
        'library_rmsd_successes': library_rmsd_successes,
        'library_rmsd_success_rate_all_attempts': rate(
            library_rmsd_successes, attempts
        ),
        'library_rmsd_success_rate_evaluable': rate(
            library_rmsd_successes, library_evaluable
        ),
        'library_both_successes': library_both_successes,
        'library_both_success_rate_all_attempts': rate(
            library_both_successes, attempts
        ),
        'library_both_success_rate_evaluable': rate(
            library_both_successes, library_evaluable
        ),
        'library_align_no_clash_evaluable': library_align_no_clash_evaluable,
        'library_align_no_clash_evaluation_errors': (
            attempts - library_align_no_clash_evaluable
            if library_requested else 0
        ),
        'library_align_no_clash_evaluable_rate': rate(
            library_align_no_clash_evaluable, attempts
        ),
        'library_align_no_clash_successes': library_align_no_clash_successes,
        'library_align_no_clash_success_rate_all_attempts': rate(
            library_align_no_clash_successes, attempts
        ),
        'library_align_no_clash_success_rate_evaluable': rate(
            library_align_no_clash_successes,
            library_align_no_clash_evaluable,
        ),
        'library_no_clash_rate_among_rmsd_successes': rate(
            library_align_no_clash_successes, library_rmsd_successes
        ),
        'library_split_conformer_only_count': library_split_only,
        'library_split_conformer_only_rate_all_attempts': rate(
            library_split_only, attempts
        ),
        'library_split_conformer_only_rate_evaluable': rate(
            library_split_only, library_evaluable
        ),
        'library_ca_only_count': library_ca_only,
        'library_ca_only_rate_all_attempts': rate(library_ca_only, attempts),
        'library_ca_only_rate_evaluable': rate(
            library_ca_only, library_evaluable
        ),
        'library_rmsd_only_count': library_rmsd_only,
        'library_rmsd_only_rate_all_attempts': rate(library_rmsd_only, attempts),
        'library_rmsd_only_rate_evaluable': rate(
            library_rmsd_only, library_evaluable
        ),
        'library_neither_count': library_neither,
        'library_neither_rate_all_attempts': rate(library_neither, attempts),
        'library_neither_rate_evaluable': rate(
            library_neither, library_evaluable
        ),
    }


def grouped_summaries(rows, keys):
    groups = {}
    for row in rows:
        group_key = tuple(str(row[key]) for key in keys)
        groups.setdefault(group_key, []).append(row)
    summaries = []
    def sort_key(group_key):
        normalized = []
        for value in group_key:
            try:
                normalized.append((0, int(value)))
            except ValueError:
                normalized.append((1, value))
        return tuple(normalized)

    for group_key in sorted(groups, key=sort_key):
        summary = dict(zip(keys, group_key))
        summary.update(summarize_rows(groups[group_key]))
        summaries.append(summary)
    return summaries


def check_expected_attempts(per_target, expected):
    if expected is None:
        return []
    warnings = []
    for group in per_target:
        if group['attempts'] != expected:
            warnings.append(
                f"{group['run_name']}/{group['target_id']}: found {group['attempts']} "
                f'attempts, expected {expected}'
            )
    return warnings


def copy_file_if_present(source, destination):
    source = Path(source)
    if not source.is_file():
        return False
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_selected_attempts(
    rows,
    jobs_by_metadata,
    copy_dir,
    include_raw_tensors,
    criterion,
):
    copy_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        job = jobs_by_metadata[row['metadata_file']]
        relative_target_dir = Path(job['relative_target_dir'])
        destination_target = copy_dir / relative_target_dir
        missing = []

        artifacts = [
            (Path(job['target_dir']) / 'target.pdb', destination_target / 'target.pdb'),
            (job['candidate_pdb'], destination_target / Path(job['candidate_pdb']).name),
            (job['candidate_sdf'], destination_target / Path(job['candidate_sdf']).name),
            (
                job['conditioning_conformer'],
                destination_target
                / 'conditioning_conformers'
                / Path(job['conditioning_conformer']).name,
            ),
            (
                job['metadata_file'],
                destination_target
                / 'raw_model_outputs'
                / Path(job['metadata_file']).name,
            ),
        ]
        if include_raw_tensors:
            raw_tensor = Path(job['metadata_file']).with_suffix('.pt')
            artifacts.append(
                (
                    raw_tensor,
                    destination_target / 'raw_model_outputs' / raw_tensor.name,
                )
            )

        library_field = COPY_LIBRARY_CONFORMER_FIELDS.get(criterion)
        if library_field is not None:
            library_conformer = Path(row[library_field])
            copied_library_conformer = (
                destination_target
                / 'matching_library_conformers'
                / f'{Path(job["metadata_file"]).stem}.sdf'
            )
            artifacts.append((library_conformer, copied_library_conformer))
            row['copied_library_conformer'] = str(copied_library_conformer)

        for source, destination in artifacts:
            if not copy_file_if_present(source, destination):
                missing.append(str(source))

        row['copied_to'] = str(destination_target)
        if missing:
            row['copy_note'] = 'Missing companion files: ' + '; '.join(missing)


def result_sort_key(row):
    sample_index = row['sample_index']
    try:
        sample_key = (0, int(sample_index))
    except (TypeError, ValueError):
        sample_key = (1, str(sample_index))
    return row['run_name'], row['target_id'], sample_key


def select_rows_for_copy(rows, criterion):
    criterion_field = COPY_CRITERION_FIELDS[criterion]
    return [row for row in rows if row[criterion_field]]


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    overall = summary['overall']
    print(f"Attempts: {overall['attempts']}")
    print(
        f"Evaluable: {overall['evaluable']}/{overall['attempts']} "
        f"({overall['evaluable_rate']:.2%})"
    )
    print(
        f"Conditioned CA success: {overall['ca_successes']}/{overall['attempts']} "
        f"({overall['ca_success_rate_all_attempts']:.2%})"
    )
    print(
        f"Conditioned 10-atom RMSD success: "
        f"{overall['rmsd_successes']}/{overall['attempts']} "
        f"({overall['rmsd_success_rate_all_attempts']:.2%})"
    )
    print(
        f"Conditioned both criteria: {overall['both_successes']}/{overall['attempts']} "
        f"({overall['both_success_rate_all_attempts']:.2%})"
    )
    if summary['library_enabled']:
        print(
            f"Library evaluable: {overall['library_evaluable']}/{overall['attempts']} "
            f"({overall['library_evaluable_rate']:.2%})"
        )
        print(
            f"Any-library CA success: "
            f"{overall['library_ca_successes']}/{overall['attempts']} "
            f"({overall['library_ca_success_rate_all_attempts']:.2%})"
        )
        print(
            f"Any-library 10-atom RMSD success: "
            f"{overall['library_rmsd_successes']}/{overall['attempts']} "
            f"({overall['library_rmsd_success_rate_all_attempts']:.2%})"
        )
        print(
            f"Any-library same-conformer both success: "
            f"{overall['library_both_successes']}/{overall['attempts']} "
            f"({overall['library_both_success_rate_all_attempts']:.2%})"
        )
        print(
            f"Any-library align + no-clash success: "
            f"{overall['library_align_no_clash_successes']}/{overall['attempts']} "
            f"({overall['library_align_no_clash_success_rate_all_attempts']:.2%})"
        )
        print(
            f"No-clash among any-library RMSD successes: "
            f"{overall['library_align_no_clash_successes']}/"
            f"{overall['library_rmsd_successes']} "
            f"({overall['library_no_clash_rate_among_rmsd_successes']:.2%})"
        )
    if summary['transfer_warnings']:
        print('Transfer warnings:')
        for warning in summary['transfer_warnings']:
            print(f'  - {warning}')


def main():
    args = parse_args()
    input_dir, output_dir, copy_dir = validate_args(args)
    metadata_paths = sorted(input_dir.rglob('raw_model_outputs/*.json'))
    if not metadata_paths:
        raise FileNotFoundError(
            f'No raw_model_outputs/*.json files found under: {input_dir}'
        )

    jobs = [load_job(path, input_dir, args) for path in metadata_paths]
    library_paths_by_run, library_sources_by_run = resolve_library_paths(jobs, args)
    for job in jobs:
        job['library_requested'] = job['run_name'] in library_paths_by_run
    workers = resolve_workers(args.workers, len(jobs))
    rows = evaluate_jobs(
        jobs,
        workers,
        args.chunksize,
        library_paths_by_run,
    )
    rows.sort(key=result_sort_key)

    selected_rows = []
    if copy_dir is not None:
        jobs_by_metadata = {job['metadata_file']: job for job in jobs}
        selected_rows = select_rows_for_copy(rows, args.copy_criterion)
        copy_selected_attempts(
            selected_rows,
            jobs_by_metadata,
            copy_dir,
            args.include_raw_tensors,
            args.copy_criterion,
        )

    per_run = grouped_summaries(rows, ('run_name',))
    per_target = grouped_summaries(rows, ('run_name', 'target_id'))
    per_length = []
    per_run_length = []
    if args.grouped_success_rates:
        per_length = grouped_summaries(rows, ('peptide_length',))
        per_run_length = grouped_summaries(rows, ('run_name', 'peptide_length'))
    summary = {
        'input': str(input_dir),
        'ca_cutoff_angstrom': args.ca_cutoff,
        'rmsd_cutoff_angstrom': args.rmsd_cutoff,
        'condition_success_definition': (
            'Compared with the sampled conditioning conformer; both_pass requires '
            'ca_pass AND rmsd_pass; both cutoffs are strict (<)'
        ),
        'library_success_definition': (
            'Compared with all capping conformers of the corresponding run; '
            'library_both_pass requires one same conformer to pass both strict cutoffs'
            if library_paths_by_run else None
        ),
        'library_align_no_clash_definition': (
            'Success requires at least one same CPcap conformer with terminal '
            '10-atom RMSD strictly below the cutoff whose merged structure has no '
            'nonlocal heavy-atom pair at or below clash_threshold times the sum '
            'of the two VDW radii; atom pairs separated by at most three bonds '
            'are excluded'
            if library_paths_by_run else None
        ),
        'success_definition': (
            'Conditioned both_pass requires ca_pass AND rmsd_pass; '
            'both cutoffs are strict (<)'
        ),
        'rate_denominator': (
            'all raw metadata attempts, including evaluation/serialization failures'
        ),
        'library_enabled': bool(library_paths_by_run),
        'library_pattern': args.library_pattern if library_paths_by_run else None,
        'clash_threshold_vdw_radius_sum_scale': (
            args.clash_threshold if library_paths_by_run else None
        ),
        'library_sources_by_run': library_sources_by_run,
        'library_sdf_counts_by_run': {
            run_name: len(paths)
            for run_name, paths in library_paths_by_run.items()
        },
        'overall': summarize_rows(rows),
        'per_run': per_run,
        'per_target': per_target,
        'grouped_success_rates_enabled': args.grouped_success_rates,
        'per_length': per_length,
        'per_run_length': per_run_length,
        'expected_attempts_per_target': args.expected_attempts_per_target,
        'transfer_warnings': check_expected_attempts(
            per_target, args.expected_attempts_per_target
        ),
        'copy_destination': str(copy_dir) if copy_dir is not None else None,
        'copy_criterion': args.copy_criterion if copy_dir is not None else None,
        'copied_attempts': len(selected_rows),
        'raw_tensors_copied': bool(copy_dir is not None and args.include_raw_tensors),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'attempt_metrics.csv'
    summary_path = output_dir / 'summary.json'
    per_run_path = output_dir / 'per_run_summary.csv'
    per_target_path = output_dir / 'per_target_summary.csv'
    write_csv(results_path, rows, RESULT_FIELDS)
    write_csv(per_run_path, per_run, per_run[0].keys())
    write_csv(per_target_path, per_target, per_target[0].keys())
    if args.grouped_success_rates:
        write_csv(
            output_dir / 'per_cyclization_summary.csv',
            per_run,
            per_run[0].keys(),
        )
        write_csv(
            output_dir / 'per_length_summary.csv',
            per_length,
            per_length[0].keys(),
        )
        write_csv(
            output_dir / 'per_cyclization_length_summary.csv',
            per_run_length,
            per_run_length[0].keys(),
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')

    if copy_dir is not None:
        write_csv(copy_dir / 'passed_manifest.csv', selected_rows, RESULT_FIELDS)
        (copy_dir / 'evaluation_summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n'
        )

    print_summary(summary)
    print(f'Reports: {output_dir}')
    if copy_dir is not None:
        print(
            f'Copied subset ({args.copy_criterion}, {len(selected_rows)} attempts): '
            f'{copy_dir}'
        )


if __name__ == '__main__':
    main()
