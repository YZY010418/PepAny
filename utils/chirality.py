#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Compatibility helpers for older checkpoints that import utils.chirality."""

from rdkit import Chem


def assign_chirality(mol, conf_id=-1):
    mol = Chem.Mol(mol)
    try:
        Chem.AssignAtomChiralTagsFromStructure(mol, confId=int(conf_id), replaceExistingTags=True)
    except Exception:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def clear_chirality(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    return mol


def get_chiral_centers(mol, include_unassigned=True):
    return Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=bool(include_unassigned),
        useLegacyImplementation=False,
    )


def copy_chirality(source_mol, target_mol, atom_map=None):
    target_mol = Chem.Mol(target_mol)
    if atom_map is None:
        atom_map = {idx: idx for idx in range(min(source_mol.GetNumAtoms(), target_mol.GetNumAtoms()))}
    for src_idx, dst_idx in atom_map.items():
        if src_idx >= source_mol.GetNumAtoms() or dst_idx >= target_mol.GetNumAtoms():
            continue
        target_mol.GetAtomWithIdx(dst_idx).SetChiralTag(
            source_mol.GetAtomWithIdx(src_idx).GetChiralTag()
        )
    return target_mol


def __getattr__(name):
    def _missing(*args, **kwargs):
        raise NotImplementedError(
            f'utils.chirality.{name} is not implemented in this checkout. '
            'The import exists only for compatibility with older checkpoints.'
        )
    _missing.__name__ = name
    return _missing
