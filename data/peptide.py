#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import Optional

from utils import register as R

from .resample import ClusterResampler
import numpy as np
import torch
from .base import BaseDataset, Summary, transform_data

# some helper functions for terminal positions
def _get_atom_coord(blk, atom_name):
    if blk is None:
        return None
    for atom in blk:
        if atom.name == atom_name:
            return np.array(atom.get_coord(), dtype=np.float32)
    return None


def _zero_term_rel_pos():
    return np.zeros(15, dtype=np.float32)


def terminal_descriptor(p1, p2, p3, p4, p5):
    """Compute direction, normal, and center of a 5-point plane"""
    direction = p1 - p5
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    points = np.array([p1, p2, p3, p4, p5])
    center = np.mean(points, axis=0)
    
    centered = points - center
    covariance = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(covariance)
    norm = eigenvectors[:, 0]
    
    ref_norm = np.cross(p1 - p2, p2 - p3)
    if np.dot(norm, ref_norm) < 0:
        norm = -norm
    
    return direction, norm, center


def _peptide_bond_amide_descriptor(block_left, block_right):
    """Descriptor for the amide plane between two adjacent residue blocks."""
    atoms = [
        _get_atom_coord(block_left, 'CA'),
        _get_atom_coord(block_left, 'C'),
        _get_atom_coord(block_left, 'O'),
        _get_atom_coord(block_right, 'N'),
        _get_atom_coord(block_right, 'CA'),
    ]
    if any(x is None for x in atoms):
        return None
    return terminal_descriptor(*atoms)


def _relative_term_feature(n_term_descriptor, c_term_descriptor):
    """Convert two terminal amide descriptors into the 15-d invariant feature."""
    try:
        d_n, norm_n, center_n = n_term_descriptor
        d_c, norm_c, center_c = c_term_descriptor
        d = np.linalg.norm(center_n - center_c)
        r = (center_n - center_c) / (d + 1e-8) if d > 1e-6 else np.zeros(3)
        
        dots = np.array([
            np.dot(r, d_n), np.dot(r, norm_n), np.dot(r, d_c), np.dot(r, norm_c),
            np.dot(d_n, d_c), np.dot(d_n, norm_c), np.dot(norm_n, d_c), np.dot(norm_n, norm_c),
        ], dtype=np.float32)
        
        triples = np.array([
            np.dot(r, np.cross(d_c, norm_c)),
            np.dot(d_n, np.cross(d_c, norm_c)),
            np.dot(norm_n, np.cross(d_c, norm_c)),
            np.dot(r, np.cross(d_n, norm_n)),
            np.dot(d_c, np.cross(d_n, norm_n)),
            np.dot(norm_c, np.cross(d_n, norm_n)),
        ], dtype=np.float32)
        
        return np.concatenate([np.array([d], dtype=np.float32), dots, triples])
    except:
        return _zero_term_rel_pos()


def compute_terminal_rel_pos(blocks):
    """Compute one condition from ACE-first and last-NME peptide-bond amide planes."""
    try:
        blocks = list(blocks)
        if len(blocks) < 2:
            return _zero_term_rel_pos()

        n_term_descriptor = _peptide_bond_amide_descriptor(blocks[0], blocks[1])
        c_term_descriptor = _peptide_bond_amide_descriptor(blocks[-2], blocks[-1])
        if n_term_descriptor is None or c_term_descriptor is None:
            return _zero_term_rel_pos()

        return _relative_term_feature(n_term_descriptor, c_term_descriptor)
    except:
        return _zero_term_rel_pos()


@R.register('PeptideDataset')
class PeptideDataset(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            cluster: Optional[str] = None,
            length_type: str = 'atom'
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    ########## Start of Overloading ##########
    def get_id(self, idx):
        idx = self.dynamic_idxs[idx]
        return self._indexes[idx][0]

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        props = self._properties[idx]
        if self.length_type == 'atom':
            return props['pocket_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['pocket_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        _id = self._indexes[idx][0]
        props = self._properties[idx]

        # get indexes (pocket + peptide)
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        cplx = self.get_raw_data(idx)
        pep_chain = props['ligand_chain_ids'][0]
        pep_block_ids = [(pep_chain, block.id) for block in cplx[pep_chain]]
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in pep_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in pep_block_ids]
        if len(pocket_block_ids) == 0: # single molecule
            center_mask = [1 for _ in pep_block_ids]

        return Summary(
            id=_id,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=props['ligand_sequences'][0], # peptide has only one chain
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=pocket_block_ids + pep_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        
        # Get base data
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)
        data = transform_data(cplx, summary.select_indexes)
        data['generate_mask'] = torch.tensor(summary.generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(summary.center_mask, dtype=torch.bool)
        
        # Add one endpoint-amide condition and copy it to every peptide block.
        pep_chain = summary.ligand_chain_ids[0]
        pep_blocks = list(cplx[pep_chain])
        
        # Pad pocket blocks with zeros
        n_pocket_blocks = len(summary.select_indexes) - len(pep_blocks)
        peptide_term_rel_pos = compute_terminal_rel_pos(pep_blocks)
        term_rel_pos = np.concatenate([
            np.zeros((n_pocket_blocks, 15), dtype=np.float32),
            np.repeat(peptide_term_rel_pos[None, :], len(pep_blocks), axis=0)
        ])
        
        data['term_rel_pos'] = torch.tensor(term_rel_pos, dtype=torch.float)
        
        # Adjust peptide position ids (existing code)
        gen_mask = data['generate_mask']
        pep_position_ids = data['position_ids'][gen_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data['position_ids'][gen_mask] = pep_position_ids
        
        return data
    

if __name__ == '__main__':
    import sys
    dataset = PeptideDataset(sys.argv[1])
    print(dataset[0])
    print(len(dataset[0]['A']))
