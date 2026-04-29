#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from Bio import PDB
from rdkit import Chem

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


PDB_TERMINAL_AMIDE_ORDER = (
    'first_ca', 'first_c', 'first_o', 'second_n', 'second_ca',
    'penultimate_ca', 'penultimate_c', 'penultimate_o', 'last_n', 'last_ca',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate terminal amide RMSD and first/last CA distance for generated PDB files.'
        )
    )
    parser.add_argument('-i', '--input', required=True, help='Generated PDB file or directory of PDB files')
    parser.add_argument('--ref', required=True, help='Reference SDF used to build the CFG condition')
    parser.add_argument(
        '--index',
        nargs='+',
        required=True,
        help=(
            'Reference atom indexes in this 10-atom order: '
            f'{",".join(PDB_TERMINAL_AMIDE_ORDER)}. '
            'Accepts comma/space separated values, optionally split with ";".'
        ),
    )
    parser.add_argument(
        '--index-base',
        type=int,
        default=0,
        help='Index base for --index, e.g. 0 for RDKit/Python indexes, 1 for 1-based atom labels',
    )
    parser.add_argument('-o', '--output', required=True, help='Output CSV path')
    parser.add_argument('--pattern', default='*.pdb', help='Generated PDB filename pattern')
    parser.add_argument('--no-recursive', action='store_true', help='Only scan the top-level input directory')
    parser.add_argument('--chain', default='L', help='Generated peptide chain ID. Default: L')
    parser.add_argument('--model-index', type=int, default=0, help='PDB model index to read. Default: 0')
    parser.add_argument('--ref-mol-index', type=int, default=0, help='Molecule index to read from --ref SDF')
    parser.add_argument(
        '--ref-ca-distance-slots',
        nargs=2,
        type=int,
        default=(0, 9),
        metavar=('SLOT_A', 'SLOT_B'),
        help=(
            'Two 0-based positions in the 10 reference-index list used for first/last CA distance. '
            'Default: 0 9 for first_ca and last_ca.'
        ),
    )
    return parser.parse_args()


def parse_atom_indexes(values, index_base):
    text = values if isinstance(values, str) else ' '.join(values)
    tokens = [tok for tok in re.split(r'[\s,;]+', text.strip()) if tok]
    if len(tokens) != 10:
        raise ValueError(f'Expected exactly 10 atom indexes, got {len(tokens)} from: {text}')
    return [int(tok) - index_base for tok in tokens]


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


def get_sdf_positions(mol, indexes):
    num_atoms = mol.GetNumAtoms()
    bad = [idx for idx in indexes if idx < 0 or idx >= num_atoms]
    if bad:
        raise ValueError(f'Atom indexes out of range for molecule with {num_atoms} atoms: {bad}')
    conf = mol.GetConformer()
    coords = []
    for idx in indexes:
        pos = conf.GetAtomPosition(int(idx))
        coords.append([pos.x, pos.y, pos.z])
    return np.asarray(coords, dtype=np.float64)


def kabsch_align(moving, target):
    moving_mean = moving.mean(axis=0)
    target_mean = target.mean(axis=0)
    moving_centered = moving - moving_mean
    target_centered = target - target_mean
    cov = moving_centered.T @ target_centered
    v, _, wt = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(wt) < 0:
        v[:, -1] *= -1
    rot = v @ wt
    return (moving - moving_mean) @ rot + target_mean


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def pair_distance(coords, slots):
    idx_a, idx_b = slots
    return float(np.linalg.norm(coords[idx_a] - coords[idx_b]))


def collect_pdb_paths(input_path, pattern, recursive):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def residue_has_peptide_backbone(residue):
    return all(atom_name in residue for atom_name in ('N', 'CA', 'C', 'O'))


def load_chain_residues(pdb_path, chain_id='L', model_index=0):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('generated', str(pdb_path))
    models = list(structure)
    if model_index < 0 or model_index >= len(models):
        raise ValueError(f'model_index {model_index} out of range for {len(models)} model(s): {pdb_path}')
    model = models[model_index]
    if chain_id not in model:
        available = ','.join(chain.id for chain in model)
        raise ValueError(f'Chain {chain_id!r} not found in {pdb_path}; available chains: {available}')
    chain = model[chain_id]
    residues = [
        residue for residue in chain
        if residue.get_resname().strip() not in {'HOH', 'WAT'} and residue_has_peptide_backbone(residue)
    ]
    if len(residues) < 2:
        raise ValueError(f'Need at least two peptide residues in chain {chain_id!r}, got {len(residues)}')
    return residues


def atom_coord(residue, atom_name):
    if atom_name not in residue:
        raise ValueError(f'Missing atom {atom_name} in residue {format_residue_id(residue)}')
    coord = residue[atom_name].get_coord()
    return np.asarray(coord, dtype=np.float64)


def format_residue_id(residue):
    _hetflag, seq_id, insertion_code = residue.id
    insertion_code = insertion_code.strip()
    suffix = insertion_code if insertion_code else ''
    return f'{residue.get_resname()}:{seq_id}{suffix}'


def pdb_terminal_amide_positions(pdb_path, chain_id='L', model_index=0):
    residues = load_chain_residues(pdb_path, chain_id=chain_id, model_index=model_index)
    first = residues[0]
    second = residues[1]
    penultimate = residues[-2]
    last = residues[-1]
    coords = [
        atom_coord(first, 'CA'),
        atom_coord(first, 'C'),
        atom_coord(first, 'O'),
        atom_coord(second, 'N'),
        atom_coord(second, 'CA'),
        atom_coord(penultimate, 'CA'),
        atom_coord(penultimate, 'C'),
        atom_coord(penultimate, 'O'),
        atom_coord(last, 'N'),
        atom_coord(last, 'CA'),
    ]
    return np.asarray(coords, dtype=np.float64), first, last


def empty_result_row(pdb_path, input_dir, chain_id):
    return {
        'file': str(Path(pdb_path).relative_to(input_dir)),
        'chain': chain_id,
        'first_residue': '',
        'last_residue': '',
        'rmsd': '',
        'n_term_rmsd': '',
        'c_term_rmsd': '',
        'ref_first_last_ca_distance': '',
        'pdb_first_last_ca_distance': '',
        'ca_distance_deviation': '',
        'ca_distance_abs_deviation': '',
        'error': '',
    }


def process_pdb_path(pdb_path, input_dir, chain_id, model_index, ref_coords, ref_ca_slots):
    row = empty_result_row(pdb_path, input_dir, chain_id)
    try:
        pdb_coords, first, last = pdb_terminal_amide_positions(
            pdb_path, chain_id=chain_id, model_index=model_index
        )
        aligned = kabsch_align(pdb_coords, ref_coords)
        terminal_rmsd = rmsd(aligned, ref_coords)
        ref_ca_distance = pair_distance(ref_coords, ref_ca_slots)
        pdb_ca_distance = float(np.linalg.norm(pdb_coords[0] - pdb_coords[9]))
        ca_distance_deviation = pdb_ca_distance - ref_ca_distance

        row['first_residue'] = format_residue_id(first)
        row['last_residue'] = format_residue_id(last)
        row['rmsd'] = f'{terminal_rmsd:.6f}'
        row['n_term_rmsd'] = f'{rmsd(aligned[:5], ref_coords[:5]):.6f}'
        row['c_term_rmsd'] = f'{rmsd(aligned[5:], ref_coords[5:]):.6f}'
        row['ref_first_last_ca_distance'] = f'{ref_ca_distance:.6f}'
        row['pdb_first_last_ca_distance'] = f'{pdb_ca_distance:.6f}'
        row['ca_distance_deviation'] = f'{ca_distance_deviation:.6f}'
        row['ca_distance_abs_deviation'] = f'{abs(ca_distance_deviation):.6f}'
    except Exception as exc:
        row['error'] = str(exc)
        row['rmsd'] = f'{math.nan}'
    return row


def main():
    args = parse_args()
    input_path = Path(args.input)
    input_base_dir = input_path.parent if input_path.is_file() else input_path
    ref_indexes = parse_atom_indexes(args.index, args.index_base)
    ref_ca_slots = validate_slot_pair(args.ref_ca_distance_slots, '--ref-ca-distance-slots')
    ref_mol = load_sdf_mol(args.ref, args.ref_mol_index)
    ref_coords = get_sdf_positions(ref_mol, ref_indexes)
    pdb_paths = collect_pdb_paths(input_path, args.pattern, recursive=not args.no_recursive)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(empty_result_row(input_base_dir / 'dummy.pdb', input_base_dir, args.chain).keys())
    with out_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        iterator = tqdm(pdb_paths, total=len(pdb_paths), desc='terminal PDB RMSD')
        for pdb_path in iterator:
            row = process_pdb_path(
                pdb_path, input_base_dir, args.chain, args.model_index, ref_coords, ref_ca_slots
            )
            writer.writerow(row)


if __name__ == '__main__':
    main()
