#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from rdkit import Chem


TERMINAL_AMIDE_ORDER = (
    'ace_ca', 'ace_c', 'ace_o', 'first_n', 'first_ca',
    'last_ca', 'last_c', 'last_o', 'nme_n', 'nme_ca',
)


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


def collect_sdf_paths(input_dir, pattern, recursive):
    input_dir = Path(input_dir)
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def main():
    args = parse_args()
    ref_mol = load_sdf_mol(args.ref, args.ref_mol_index)
    if args.index is None:
        ref_indexes = infer_terminal_amide_indexes(ref_mol)
    else:
        ref_indexes = parse_atom_indexes(args.index, args.index_base)
    ref_coords = get_positions(ref_mol, ref_indexes)

    gen_indexes = None
    if args.gen_index is not None:
        gen_indexes = parse_atom_indexes(args.gen_index, args.gen_index_base)

    sdf_paths = collect_sdf_paths(args.input, args.pattern, recursive=not args.no_recursive)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['file', 'rmsd', 'n_term_rmsd', 'c_term_rmsd', 'error']
        )
        writer.writeheader()
        for sdf_path in sdf_paths:
            row = {
                'file': str(sdf_path.relative_to(args.input)),
                'rmsd': '',
                'n_term_rmsd': '',
                'c_term_rmsd': '',
                'error': '',
            }
            try:
                mol = load_sdf_mol(sdf_path, args.mol_index)
                cur_indexes = infer_generated_indexes(mol) if gen_indexes is None else gen_indexes
                gen_coords = get_positions(mol, cur_indexes)
                aligned = kabsch_align(gen_coords, ref_coords)
                row['rmsd'] = f'{rmsd(aligned, ref_coords):.6f}'
                row['n_term_rmsd'] = f'{rmsd(aligned[:5], ref_coords[:5]):.6f}'
                row['c_term_rmsd'] = f'{rmsd(aligned[5:], ref_coords[5:]):.6f}'
            except Exception as exc:
                row['error'] = str(exc)
                row['rmsd'] = f'{math.nan}'
            writer.writerow(row)


if __name__ == '__main__':
    main()
