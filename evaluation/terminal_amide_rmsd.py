#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import csv
import math
from multiprocessing import Pool, cpu_count
import re
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


TERMINAL_AMIDE_ORDER = (
    'ace_ca', 'ace_c', 'ace_o', 'first_n', 'first_ca',
    'last_ca', 'last_c', 'last_o', 'nme_n', 'nme_ca',
)

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'P': 1.80, 'S': 1.80,
    'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'I': 1.98,
}

WORKER_REF_MOL = None
WORKER_REF_PATH = None
WORKER_REF_INDEXES = None
WORKER_REF_COORDS = None
WORKER_INPUT_DIR = None
WORKER_GEN_INDEXES = None
WORKER_REF_CA_DISTANCE_SLOTS = (0, 9)
WORKER_GEN_CA_DISTANCE_SLOTS = (0, 9)
WORKER_MOL_INDEX = 0
WORKER_MERGE = False
WORKER_MERGE_OUTPUT_DIR = None
WORKER_MERGE_RMSD_CUTOFF = 0.5
WORKER_CLASH_THRESHOLD = 0.8
WORKER_NO_CLASH_CHECK = False
WORKER_COMPLEX_PDB_DIR = None
WORKER_REQUIRE_RECEPTOR_CLASH_CHECK = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RMSD between generated terminal amides and a reference CFG SDF.'
    )
    parser.add_argument('-i', '--input', required=True, help='Directory containing generated SDF files')
    parser.add_argument('--ref', required=True, help='Reference/capping SDF used to build the CFG condition')
    parser.add_argument(
        '--index',
        nargs='+',
        default=None,
        help=(
            'Reference atom indexes used for CFG. Default order/mode: '
            f'{",".join(TERMINAL_AMIDE_ORDER)}. '
            'If omitted, infer this order from the reference SDF when possible. '
            'Accepts comma/space separated values, optionally split with ";".'
        )
    )
    parser.add_argument(
        '--index-base',
        type=int,
        default=0,
        help='Index base for --index, e.g. 0 for RDKit/Python indexes, 1 for 1-based atom labels'
    )
    parser.add_argument(
        '--gen-index',
        nargs='+',
        default=None,
        help=(
            'Optional generated atom indexes in the same default 10-atom order: '
            f'{",".join(TERMINAL_AMIDE_ORDER)}. '
            'If omitted, infer indexes for DeNovoCappedLinearPeptide outputs.'
        )
    )
    parser.add_argument(
        '--gen-index-base',
        type=int,
        default=0,
        help='Index base for --gen-index'
    )
    parser.add_argument('-o', '--output', required=True, help='Output CSV path')
    parser.add_argument('--pattern', default='*.sdf', help='Generated SDF filename pattern')
    parser.add_argument('--no-recursive', action='store_true', help='Only scan the top-level input directory')
    parser.add_argument('--ref-mol-index', type=int, default=0, help='Molecule index to read from --ref SDF')
    parser.add_argument('--mol-index', type=int, default=0, help='Molecule index to read from generated SDF files')
    parser.add_argument(
        '--merge',
        action='store_true',
        help=(
            'For generated/reference pairs with terminal RMSD below --merge-rmsd-cutoff, '
            'align the reference cap, merge it into the generated peptide, and save '
            'clash-free merged SDF files.'
        )
    )
    parser.add_argument(
        '--merge-output-dir',
        default=None,
        help='Directory for merged SDF files. Default: <output CSV stem>_merged'
    )
    parser.add_argument(
        '--merge-rmsd-cutoff',
        type=float,
        default=0.5,
        help='Maximum terminal RMSD for attempting merge. Default: 0.5'
    )
    parser.add_argument(
        '--clash-threshold',
        type=float,
        default=0.8,
        help='VDW radius scaling threshold for clash rejection. Default follows CPSea2: 0.8'
    )
    parser.add_argument(
        '--no-clash-check',
        action='store_true',
        help='Write merged structures without rejecting VDW clashes'
    )
    parser.add_argument(
        '--complex-pdb-dir',
        default=None,
        help=(
            'Optional directory containing generated complex PDB files with the same '
            'relative names as input SDFs. If omitted, a sibling .pdb next to each '
            'SDF is used when present for receptor clash checks.'
        )
    )
    parser.add_argument(
        '--require-receptor-clash-check',
        action='store_true',
        help='Fail merge attempts when no matching complex PDB is available for receptor clash checks'
    )
    parser.add_argument(
        '-p',
        '--processors',
        type=int,
        default=1,
        help='Number of worker processes. Use 0 for all CPUs. Default: 1'
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=1,
        help='Number of SDF files sent to each worker at a time. Default: 1'
    )
    parser.add_argument(
        '--ref-ca-distance-slots',
        nargs=2,
        type=int,
        default=(0, 9),
        metavar=('SLOT_A', 'SLOT_B'),
        help=(
            'Two 0-based positions in the 10 reference terminal-index list used '
            'for the ACE.CA/NME.CA distance. Default: 0 9 for normal order. '
            'For crossed reference order, usually use: 4 5.'
        )
    )
    parser.add_argument(
        '--gen-ca-distance-slots',
        nargs=2,
        type=int,
        default=(0, 9),
        metavar=('SLOT_A', 'SLOT_B'),
        help='Two 0-based positions in the 10 generated terminal-index list. Default: 0 9.'
    )
    return parser.parse_args()


def parse_atom_indexes(values, index_base):
    if isinstance(values, str):
        text = values
    else:
        text = ' '.join(values)
    tokens = [tok for tok in re.split(r'[\s,;]+', text.strip()) if tok]
    if len(tokens) != 10:
        raise ValueError(f'Expected exactly 10 atom indexes, got {len(tokens)} from: {text}')
    return [int(tok) - index_base for tok in tokens]


def load_sdf_mol(path, mol_index=0):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    if supplier is None:
        raise ValueError(f'Could not open SDF: {path}')
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        raise ValueError(f'No valid molecule found in SDF: {path}')
    if mol_index < 0 or mol_index >= len(mols):
        raise ValueError(f'mol_index {mol_index} out of range for {len(mols)} molecule(s): {path}')
    mol = mols[mol_index]
    if mol.GetNumConformers() == 0:
        raise ValueError(f'Molecule has no 3D conformer: {path}')
    return mol


def get_positions(mol, indexes):
    num_atoms = mol.GetNumAtoms()
    bad = [idx for idx in indexes if idx < 0 or idx >= num_atoms]
    if bad:
        raise ValueError(f'Atom indexes out of range for molecule with {num_atoms} atoms: {bad}')
    conf = mol.GetConformer()
    coords = []
    for idx in indexes:
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
    return np.asarray(coords, dtype=np.float64)


def atom_symbol(mol, idx):
    return mol.GetAtomWithIdx(idx).GetSymbol()


def neighbors(mol, idx, symbol=None, exclude=None):
    exclude = set() if exclude is None else set(exclude)
    hits = []
    atom = mol.GetAtomWithIdx(idx)
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in exclude:
            continue
        if symbol is not None and nbr.GetSymbol() != symbol:
            continue
        hits.append(nbr_idx)
    return hits


def choose_one(candidates, label):
    if len(candidates) != 1:
        raise ValueError(f'Could not uniquely infer {label}: candidates={candidates}')
    return candidates[0]


def infer_terminal_amide_indexes(mol):
    """Infer indexes in TERMINAL_AMIDE_ORDER for DeNovoCappedLinearPeptide SDF atom order."""
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 10:
        raise ValueError(f'Need at least 10 atoms to infer capped peptide termini, got {num_atoms}')

    # DeNovoCappedLinearPeptide writes ACE first: [ACE.CA, ACE.C, ACE.O],
    # then peptide residues in canonical order, and NME last: [NME.N, NME.CA].
    n_term = [0, 1, 2, 3, 4]
    nme_n = num_atoms - 2
    nme_ca = num_atoms - 1
    if atom_symbol(mol, nme_n) != 'N' or atom_symbol(mol, nme_ca) != 'C':
        raise ValueError(
            'Could not infer NME atoms from the last two SDF atoms. '
            'Pass --index/--gen-index explicitly.'
        )

    last_c = choose_one(
        neighbors(mol, nme_n, symbol='C', exclude={nme_ca}),
        'last residue carbonyl C bonded to NME.N'
    )
    last_o_candidates = neighbors(mol, last_c, symbol='O', exclude={nme_n})
    double_o_candidates = [
        idx for idx in last_o_candidates
        if mol.GetBondBetweenAtoms(last_c, idx).GetBondType() == Chem.rdchem.BondType.DOUBLE
    ]
    last_o = choose_one(double_o_candidates or last_o_candidates, 'last residue carbonyl O')

    ca_candidates = neighbors(mol, last_c, symbol='C', exclude={nme_n, last_o})
    if last_c - 1 in ca_candidates:
        last_ca = last_c - 1
    else:
        last_ca = choose_one(ca_candidates, 'last residue CA')

    c_term = [last_ca, last_c, last_o, nme_n, nme_ca]
    return n_term + c_term


def infer_generated_indexes(mol):
    return infer_terminal_amide_indexes(mol)


def kabsch_align(moving, target):
    rot, moving_mean, target_mean = kabsch_transform(moving, target)
    return (moving - moving_mean) @ rot + target_mean


def kabsch_transform(moving, target):
    moving_mean = moving.mean(axis=0)
    target_mean = target.mean(axis=0)
    moving_centered = moving - moving_mean
    target_centered = target - target_mean
    cov = moving_centered.T @ target_centered
    v, _, wt = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(wt) < 0:
        v[:, -1] *= -1
    rot = v @ wt
    return rot, moving_mean, target_mean


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def validate_slot_pair(slots, label):
    slots = tuple(int(slot) for slot in slots)
    if len(slots) != 2:
        raise ValueError(f'{label} must contain exactly two slots')
    bad = [slot for slot in slots if slot < 0 or slot >= 10]
    if bad:
        raise ValueError(f'{label} slots must be in [0, 9], got {bad}')
    if slots[0] == slots[1]:
        raise ValueError(f'{label} slots must refer to two different atoms')
    return slots


def pair_distance(coords, slots):
    idx_a, idx_b = slots
    return float(np.linalg.norm(coords[idx_a] - coords[idx_b]))


def collect_sdf_paths(input_dir, pattern, recursive):
    input_dir = Path(input_dir)
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def all_positions(mol):
    conf = mol.GetConformer()
    coords = []
    for idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
    return np.asarray(coords, dtype=np.float64)


def set_atom_position(mol, idx, coord):
    mol.GetConformer().SetAtomPosition(
        int(idx),
        Point3D(float(coord[0]), float(coord[1]), float(coord[2]))
    )


def align_ref_to_generated(ref_mol, ref_indexes, gen_mol, gen_indexes):
    ref_work = Chem.Mol(ref_mol)
    gen_work = Chem.Mol(gen_mol)

    ref_anchor = get_positions(ref_work, ref_indexes)
    gen_anchor = get_positions(gen_work, gen_indexes)
    rot, ref_mean, gen_mean = kabsch_transform(ref_anchor, gen_anchor)
    aligned_ref_coords = (all_positions(ref_work) - ref_mean) @ rot + gen_mean
    for idx, coord in enumerate(aligned_ref_coords):
        set_atom_position(ref_work, idx, coord)

    # CPSea2 averages the matched anchor atoms after fitting; keep that behavior
    # so the new terminal bonds land midway between the two accepted geometries.
    fitted_ref_anchor = get_positions(ref_work, ref_indexes)
    for gen_idx, ref_idx, ref_coord in zip(gen_indexes, ref_indexes, fitted_ref_anchor):
        gen_pos = gen_work.GetConformer().GetAtomPosition(int(gen_idx))
        gen_coord = np.asarray([gen_pos.x, gen_pos.y, gen_pos.z], dtype=np.float64)
        averaged = (gen_coord + ref_coord) * 0.5
        set_atom_position(gen_work, gen_idx, averaged)
        set_atom_position(ref_work, ref_idx, averaged)

    return ref_work, gen_work


def remove_atoms_with_map(mol, remove_indexes):
    remove_indexes = {int(idx) for idx in remove_indexes}
    bad = [idx for idx in remove_indexes if idx < 0 or idx >= mol.GetNumAtoms()]
    if bad:
        raise ValueError(f'Cannot remove atom indexes out of range: {bad}')

    old_to_new = {}
    new_idx = 0
    for old_idx in range(mol.GetNumAtoms()):
        if old_idx in remove_indexes:
            continue
        old_to_new[old_idx] = new_idx
        new_idx += 1

    rw_mol = Chem.RWMol(mol)
    for idx in sorted(remove_indexes, reverse=True):
        rw_mol.RemoveAtom(idx)
    return rw_mol.GetMol(), old_to_new


def heavy_atom_mol_with_map(mol):
    remove_indexes = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol().upper() in {'H', 'D'}
    ]
    if not remove_indexes:
        return Chem.Mol(mol), {idx: idx for idx in range(mol.GetNumAtoms())}
    return remove_atoms_with_map(mol, remove_indexes)


def remap_indexes(indexes, old_to_new, label):
    remapped = []
    for idx in indexes:
        idx = int(idx)
        if idx not in old_to_new:
            raise ValueError(f'{label} atom index {idx} is hydrogen/deuterium and cannot be used for terminal merge')
        remapped.append(old_to_new[idx])
    return remapped


def merge_terminal_cap(ref_mol, ref_indexes, gen_mol, gen_indexes):
    """Merge an aligned reference cap into a DeNovoCappedLinearPeptide output.

    The two 5-atom groups are interpreted in comparison order:
    first group maps to the generated N side and second group maps to the
    generated C side. Each group must keep the standard order
    [carbonyl_CA, carbonyl_C, carbonyl_O, amide_N, amide_CA].
    """
    ref_mol, ref_old_to_new = heavy_atom_mol_with_map(ref_mol)
    gen_mol, gen_old_to_new = heavy_atom_mol_with_map(gen_mol)
    ref_indexes = remap_indexes(ref_indexes, ref_old_to_new, 'Reference')
    gen_indexes = remap_indexes(gen_indexes, gen_old_to_new, 'Generated')

    ref_to_gen_n = list(ref_indexes[:5])
    ref_to_gen_c = list(ref_indexes[5:])
    gen_n = list(gen_indexes[:5])
    gen_c = list(gen_indexes[5:])

    ref_work, gen_work = align_ref_to_generated(ref_mol, ref_indexes, gen_mol, gen_indexes)

    # Remove placeholder ACE/NME atoms from the generated peptide, then remove
    # the overlapping half of each reference terminal group.
    gen_remove = set(gen_n[:3] + gen_c[3:])
    ref_remove = set(ref_to_gen_n[3:] + ref_to_gen_c[:3])

    gen_kept, gen_map = remove_atoms_with_map(gen_work, gen_remove)
    ref_kept, ref_map = remove_atoms_with_map(ref_work, ref_remove)

    combined = Chem.CombineMols(gen_kept, ref_kept)
    rw_merged = Chem.RWMol(combined)
    ref_offset = gen_kept.GetNumAtoms()

    new_bonds = [
        (gen_n[3], ref_to_gen_n[1]),  # generated first-residue N to reference carbonyl C
        (gen_c[1], ref_to_gen_c[3]),  # generated last-residue C to reference amide N
    ]
    for gen_idx, ref_idx in new_bonds:
        if gen_idx not in gen_map:
            raise ValueError(f'Generated link atom {gen_idx} was removed unexpectedly')
        if ref_idx not in ref_map:
            raise ValueError(f'Reference link atom {ref_idx} was removed unexpectedly')
        rw_merged.AddBond(
            int(gen_map[gen_idx]),
            int(ref_offset + ref_map[ref_idx]),
            Chem.rdchem.BondType.SINGLE,
        )

    merged = rw_merged.GetMol()
    try:
        Chem.SanitizeMol(merged)
    except Exception:
        pass
    return merged


def vdw_radius(symbol):
    return VDW_RADII.get(str(symbol).upper(), 1.80)


def detect_clashes(mol, threshold=0.8):
    conf = mol.GetConformer()
    heavy = [
        idx for idx in range(mol.GetNumAtoms())
        if atom_symbol(mol, idx).upper() not in {'H', 'D'}
    ]
    topo_dist = Chem.GetDistanceMatrix(mol)
    clashes = []
    for left_pos, i in enumerate(heavy):
        coord_i = conf.GetAtomPosition(i)
        xyz_i = np.asarray([coord_i.x, coord_i.y, coord_i.z], dtype=np.float64)
        sym_i = atom_symbol(mol, i)
        for j in heavy[left_pos + 1:]:
            path_len = topo_dist[i, j]
            if 0 < path_len <= 3:
                continue
            coord_j = conf.GetAtomPosition(j)
            xyz_j = np.asarray([coord_j.x, coord_j.y, coord_j.z], dtype=np.float64)
            dist = float(np.linalg.norm(xyz_i - xyz_j))
            cutoff = float(threshold) * (vdw_radius(sym_i) + vdw_radius(atom_symbol(mol, j)))
            if dist <= cutoff:
                clashes.append((i, j, dist, cutoff))
    return clashes


def infer_pdb_element(line):
    element = line[76:78].strip()
    if element:
        return element.upper()
    atom_name = line[12:16].strip()
    atom_name = re.sub(r'[^A-Za-z]', '', atom_name)
    if len(atom_name) >= 2 and atom_name[:2].upper() in VDW_RADII:
        return atom_name[:2].upper()
    return atom_name[:1].upper()


def read_pdb_heavy_atoms(path):
    atoms = []
    with open(path) as handle:
        for line in handle:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            element = infer_pdb_element(line)
            if element in {'H', 'D'}:
                continue
            try:
                coord = np.asarray([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ], dtype=np.float64)
            except ValueError:
                continue
            atoms.append((element, coord))
    return atoms


def heavy_atom_coords_by_symbol(mol):
    conf = mol.GetConformer()
    coords = {}
    for idx in range(mol.GetNumAtoms()):
        symbol = atom_symbol(mol, idx).upper()
        if symbol in {'H', 'D'}:
            continue
        pos = conf.GetAtomPosition(idx)
        coords.setdefault(symbol, []).append(np.asarray([pos.x, pos.y, pos.z], dtype=np.float64))
    return coords


def matching_complex_pdb_path(sdf_path, input_dir, complex_pdb_dir):
    if complex_pdb_dir is None:
        candidate = sdf_path.with_suffix('.pdb')
    else:
        candidate = Path(complex_pdb_dir) / sdf_path.relative_to(input_dir).with_suffix('.pdb')
    return candidate if candidate.exists() else None


def receptor_atoms_from_complex_pdb(pdb_path, gen_mol, match_tolerance=0.05):
    pdb_atoms = read_pdb_heavy_atoms(pdb_path)
    if not pdb_atoms:
        raise ValueError(f'No heavy atoms found in complex PDB: {pdb_path}')

    gen_coords_by_symbol = heavy_atom_coords_by_symbol(gen_mol)
    gen_heavy_count = sum(len(coords) for coords in gen_coords_by_symbol.values())
    receptor_atoms = []
    matched_ligand_atoms = 0
    for element, coord in pdb_atoms:
        candidates = gen_coords_by_symbol.get(element.upper(), [])
        is_ligand = any(np.linalg.norm(coord - gen_coord) <= match_tolerance for gen_coord in candidates)
        if is_ligand:
            matched_ligand_atoms += 1
        else:
            receptor_atoms.append((element, coord))

    min_expected = max(1, int(0.5 * gen_heavy_count))
    if matched_ligand_atoms < min_expected:
        raise ValueError(
            f'Only matched {matched_ligand_atoms}/{gen_heavy_count} generated ligand heavy atoms '
            f'in complex PDB {pdb_path}'
        )
    return receptor_atoms


def detect_receptor_clashes(mol, receptor_atoms, threshold=0.8):
    conf = mol.GetConformer()
    clashes = []
    ligand_heavy = [
        idx for idx in range(mol.GetNumAtoms())
        if atom_symbol(mol, idx).upper() not in {'H', 'D'}
    ]
    for i in ligand_heavy:
        coord_i = conf.GetAtomPosition(i)
        xyz_i = np.asarray([coord_i.x, coord_i.y, coord_i.z], dtype=np.float64)
        sym_i = atom_symbol(mol, i)
        for receptor_idx, (receptor_symbol, receptor_coord) in enumerate(receptor_atoms):
            dist = float(np.linalg.norm(xyz_i - receptor_coord))
            cutoff = float(threshold) * (vdw_radius(sym_i) + vdw_radius(receptor_symbol))
            if dist <= cutoff:
                clashes.append((i, receptor_idx, dist, cutoff))
    return clashes


def merged_output_path(output_dir, input_dir, sdf_path):
    rel = sdf_path.relative_to(input_dir).with_suffix('')
    stem = '__'.join(rel.parts)
    return Path(output_dir) / f'{stem}_merged.sdf'


def write_sdf(mol, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def init_worker(
        ref_path,
        ref_mol_index,
        ref_indexes,
        input_dir,
        gen_indexes,
        ref_ca_distance_slots,
        gen_ca_distance_slots,
        mol_index,
        merge,
        merge_output_dir,
        merge_rmsd_cutoff,
        clash_threshold,
        no_clash_check,
        complex_pdb_dir,
        require_receptor_clash_check,
    ):
    global WORKER_REF_MOL
    global WORKER_REF_PATH
    global WORKER_REF_INDEXES
    global WORKER_REF_COORDS
    global WORKER_INPUT_DIR
    global WORKER_GEN_INDEXES
    global WORKER_REF_CA_DISTANCE_SLOTS
    global WORKER_GEN_CA_DISTANCE_SLOTS
    global WORKER_MOL_INDEX
    global WORKER_MERGE
    global WORKER_MERGE_OUTPUT_DIR
    global WORKER_MERGE_RMSD_CUTOFF
    global WORKER_CLASH_THRESHOLD
    global WORKER_NO_CLASH_CHECK
    global WORKER_COMPLEX_PDB_DIR
    global WORKER_REQUIRE_RECEPTOR_CLASH_CHECK

    WORKER_REF_MOL = load_sdf_mol(ref_path, ref_mol_index)
    WORKER_REF_PATH = str(ref_path)
    WORKER_REF_INDEXES = list(ref_indexes)
    WORKER_REF_COORDS = get_positions(WORKER_REF_MOL, WORKER_REF_INDEXES)
    WORKER_INPUT_DIR = Path(input_dir)
    WORKER_GEN_INDEXES = None if gen_indexes is None else list(gen_indexes)
    WORKER_REF_CA_DISTANCE_SLOTS = validate_slot_pair(
        ref_ca_distance_slots, '--ref-ca-distance-slots'
    )
    WORKER_GEN_CA_DISTANCE_SLOTS = validate_slot_pair(
        gen_ca_distance_slots, '--gen-ca-distance-slots'
    )
    WORKER_MOL_INDEX = int(mol_index)
    WORKER_MERGE = bool(merge)
    WORKER_MERGE_OUTPUT_DIR = None if merge_output_dir is None else Path(merge_output_dir)
    WORKER_MERGE_RMSD_CUTOFF = float(merge_rmsd_cutoff)
    WORKER_CLASH_THRESHOLD = float(clash_threshold)
    WORKER_NO_CLASH_CHECK = bool(no_clash_check)
    WORKER_COMPLEX_PDB_DIR = None if complex_pdb_dir is None else Path(complex_pdb_dir)
    WORKER_REQUIRE_RECEPTOR_CLASH_CHECK = bool(require_receptor_clash_check)


def merge_csv_fieldnames(merge):
    fieldnames = [
        'file', 'rmsd', 'n_term_rmsd', 'c_term_rmsd',
        'ref_nme_ace_ca_distance', 'gen_nme_ace_ca_distance',
        'nme_ace_ca_distance_deviation', 'nme_ace_ca_distance_abs_deviation',
        'error'
    ]
    if merge:
        fieldnames.extend([
            'merge_status', 'merge_file', 'self_clash_count',
            'receptor_clash_count', 'receptor_clash_note', 'merge_error'
        ])
    return fieldnames


def empty_result_row(sdf_path):
    sdf_path = Path(sdf_path)
    row = {
        'file': str(sdf_path.relative_to(WORKER_INPUT_DIR)),
        'rmsd': '',
        'n_term_rmsd': '',
        'c_term_rmsd': '',
        'ref_nme_ace_ca_distance': '',
        'gen_nme_ace_ca_distance': '',
        'nme_ace_ca_distance_deviation': '',
        'nme_ace_ca_distance_abs_deviation': '',
        'error': '',
    }
    if WORKER_MERGE:
        row.update({
            'merge_status': '',
            'merge_file': '',
            'self_clash_count': '',
            'receptor_clash_count': '',
            'receptor_clash_note': '',
            'merge_error': '',
        })
    return row


def process_sdf_path(sdf_path):
    sdf_path = Path(sdf_path)
    row = empty_result_row(sdf_path)
    try:
        mol = load_sdf_mol(sdf_path, WORKER_MOL_INDEX)
        cur_indexes = (
            infer_generated_indexes(mol)
            if WORKER_GEN_INDEXES is None
            else WORKER_GEN_INDEXES
        )
        gen_coords = get_positions(mol, cur_indexes)
        aligned = kabsch_align(gen_coords, WORKER_REF_COORDS)
        terminal_rmsd = rmsd(aligned, WORKER_REF_COORDS)
        row['rmsd'] = f'{terminal_rmsd:.6f}'
        row['n_term_rmsd'] = f'{rmsd(aligned[:5], WORKER_REF_COORDS[:5]):.6f}'
        row['c_term_rmsd'] = f'{rmsd(aligned[5:], WORKER_REF_COORDS[5:]):.6f}'
        ref_ca_distance = pair_distance(WORKER_REF_COORDS, WORKER_REF_CA_DISTANCE_SLOTS)
        gen_ca_distance = pair_distance(gen_coords, WORKER_GEN_CA_DISTANCE_SLOTS)
        ca_distance_deviation = gen_ca_distance - ref_ca_distance
        row['ref_nme_ace_ca_distance'] = f'{ref_ca_distance:.6f}'
        row['gen_nme_ace_ca_distance'] = f'{gen_ca_distance:.6f}'
        row['nme_ace_ca_distance_deviation'] = f'{ca_distance_deviation:.6f}'
        row['nme_ace_ca_distance_abs_deviation'] = f'{abs(ca_distance_deviation):.6f}'
        if WORKER_MERGE:
            update_merge_row(row, sdf_path, mol, cur_indexes, terminal_rmsd)
    except Exception as exc:
        row['error'] = str(exc)
        row['rmsd'] = f'{math.nan}'
        if WORKER_MERGE:
            row['merge_status'] = 'error'
    return row


def update_merge_row(row, sdf_path, mol, cur_indexes, terminal_rmsd):
    if terminal_rmsd >= WORKER_MERGE_RMSD_CUTOFF:
        row['merge_status'] = 'rmsd_cutoff'
        return

    try:
        merged = merge_terminal_cap(WORKER_REF_MOL, WORKER_REF_INDEXES, mol, cur_indexes)
        self_clashes = []
        receptor_clashes = []
        if not WORKER_NO_CLASH_CHECK:
            self_clashes = detect_clashes(merged, threshold=WORKER_CLASH_THRESHOLD)
            pdb_path = matching_complex_pdb_path(
                sdf_path, WORKER_INPUT_DIR, WORKER_COMPLEX_PDB_DIR
            )
            if pdb_path is None:
                row['receptor_clash_note'] = 'no_complex_pdb'
                if WORKER_REQUIRE_RECEPTOR_CLASH_CHECK:
                    raise ValueError('No matching complex PDB for receptor clash check')
            else:
                receptor_atoms = receptor_atoms_from_complex_pdb(pdb_path, mol)
                receptor_clashes = detect_receptor_clashes(
                    merged, receptor_atoms, threshold=WORKER_CLASH_THRESHOLD
                )
        row['self_clash_count'] = str(len(self_clashes))
        row['receptor_clash_count'] = str(len(receptor_clashes))
        if self_clashes or receptor_clashes:
            row['merge_status'] = 'clash'
        else:
            merge_path = merged_output_path(
                WORKER_MERGE_OUTPUT_DIR, WORKER_INPUT_DIR, sdf_path
            )
            merged.SetProp('terminal_amide_rmsd', row['rmsd'])
            merged.SetProp('source_sdf', str(sdf_path))
            merged.SetProp('reference_sdf', WORKER_REF_PATH)
            write_sdf(merged, merge_path)
            row['merge_status'] = 'saved'
            row['merge_file'] = str(merge_path)
    except Exception as merge_exc:
        row['merge_status'] = 'merge_error'
        row['merge_error'] = str(merge_exc)


def resolve_processors(processors, n_items):
    if processors < 0:
        raise ValueError('--processors must be >= 0')
    if processors == 0:
        processors = cpu_count()
    if n_items == 0:
        return 1
    return max(1, min(int(processors), int(n_items)))


def iter_result_rows(sdf_paths, args, ref_indexes, gen_indexes, merge_output_dir):
    worker_args = (
        str(args.ref),
        args.ref_mol_index,
        ref_indexes,
        str(args.input),
        gen_indexes,
        args.ref_ca_distance_slots,
        args.gen_ca_distance_slots,
        args.mol_index,
        args.merge,
        None if merge_output_dir is None else str(merge_output_dir),
        args.merge_rmsd_cutoff,
        args.clash_threshold,
        args.no_clash_check,
        args.complex_pdb_dir,
        args.require_receptor_clash_check,
    )
    processors = resolve_processors(args.processors, len(sdf_paths))
    chunksize = max(1, int(args.chunksize))
    path_strings = [str(path) for path in sdf_paths]

    if processors == 1:
        init_worker(*worker_args)
        iterator = map(process_sdf_path, path_strings)
        yield from tqdm(iterator, total=len(path_strings), desc='terminal RMSD')
        return

    with Pool(processes=processors, initializer=init_worker, initargs=worker_args) as pool:
        iterator = pool.imap(process_sdf_path, path_strings, chunksize=chunksize)
        yield from tqdm(iterator, total=len(path_strings), desc='terminal RMSD')


def main():
    args = parse_args()
    args.ref_ca_distance_slots = validate_slot_pair(
        args.ref_ca_distance_slots, '--ref-ca-distance-slots'
    )
    args.gen_ca_distance_slots = validate_slot_pair(
        args.gen_ca_distance_slots, '--gen-ca-distance-slots'
    )
    input_dir = Path(args.input)
    ref_mol = load_sdf_mol(args.ref, args.ref_mol_index)
    if args.index is None:
        ref_indexes = infer_terminal_amide_indexes(ref_mol)
    else:
        ref_indexes = parse_atom_indexes(args.index, args.index_base)

    gen_indexes = None
    if args.gen_index is not None:
        gen_indexes = parse_atom_indexes(args.gen_index, args.gen_index_base)

    sdf_paths = collect_sdf_paths(input_dir, args.pattern, recursive=not args.no_recursive)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merge_output_dir = None
    if args.merge:
        merge_output_dir = (
            Path(args.merge_output_dir)
            if args.merge_output_dir is not None
            else out_path.with_suffix('').parent / f'{out_path.stem}_merged'
        )
        merge_output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = merge_csv_fieldnames(args.merge)

    with out_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        rows = iter_result_rows(sdf_paths, args, ref_indexes, gen_indexes, merge_output_dir)
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    main()
