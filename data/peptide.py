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

def compute_term_rel_pos(block_n, block, block_c):
    """Compute 15-dim terminal relative position feature"""
    try:
        def get_atom_coord(blk, atom_name):
            for atom in blk:
                if atom.name == atom_name:
                    return np.array(atom.get_coord(), dtype=np.float32)
            return None
        
        c_ac = get_atom_coord(block_n, 'C') if block_n else None
        c_n = get_atom_coord(block, 'C')
        o_n = get_atom_coord(block, 'O')
        n_n = get_atom_coord(block, 'N')
        ca_n = get_atom_coord(block, 'CA')
        
        ca_c = get_atom_coord(block, 'CA')
        c_c = get_atom_coord(block, 'C')
        o_c = get_atom_coord(block, 'O')
        n_c = get_atom_coord(block_c, 'N') if block_c else None
        ca_cn = get_atom_coord(block_c, 'CA') if block_c else None
        
        if any(x is None for x in [c_n, o_n, n_n, ca_n, c_c, o_c]):
            return np.zeros(15, dtype=np.float32)
        
        if c_ac is not None:
            d_n, norm_n, center_n = terminal_descriptor(c_ac, c_n, o_n, n_n, ca_n)
        else:
            d_n, norm_n, center_n = np.zeros(3), np.zeros(3), get_atom_coord(block, 'CA') # make the process smooth
        
        if ca_cn is not None:
            d_c, norm_c, center_c = terminal_descriptor(ca_c, c_c, o_c, n_c, ca_cn)
        else:
            d_c, norm_c, center_c = np.zeros(3), np.zeros(3), get_atom_coord(block, 'CA')
        
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
        return np.zeros(15, dtype=np.float32)


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
        
        # Add term_rel_pos for peptide blocks (last part after pocket)
        pep_chain = summary.ligand_chain_ids[0]
        pep_blocks = cplx[pep_chain]
        term_pos_list = []
        for i, block in enumerate(pep_blocks):
            block_n = pep_blocks[i-1] if i > 0 else None
            block_c = pep_blocks[i+1] if i < len(pep_blocks)-1 else None
            term_pos_list.append(compute_term_rel_pos(block_n, block, block_c))
        
        # Pad pocket blocks with zeros
        n_pocket_blocks = len(summary.select_indexes) - len(pep_blocks)
        term_rel_pos = np.concatenate([
            np.zeros((n_pocket_blocks, 15), dtype=np.float32),
            np.array(term_pos_list, dtype=np.float32)
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