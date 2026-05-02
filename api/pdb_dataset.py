#!/usr/bin/python
# -*- coding:utf-8 -*-
import inspect
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.interface import compute_pocket
from data.base import transform_data
from data.peptide import compute_terminal_rel_pos

from .templates import BaseTemplate, ComplexDesc


def _pdb_to_complex_supports_preserve_residue_names():
    try:
        return 'preserve_residue_names' in inspect.signature(pdb_to_complex).parameters
    except (TypeError, ValueError):
        return False


class PDBDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            pdb_paths: List[str] = None,
            tgt_chains: List[List[str]] = None,
            lig_chains: List[List[str]] = None,
            template_config: BaseTemplate = None,
            n_samples: int = 1,
            pdb_dir: str = None,
            pdb_pattern: str = '*.pdb',
            recursive: bool = False,
            ligand_size_mode: str = 'ca',
        ):
        super().__init__()
        self.config = template_config
        self.pdb_paths = self._resolve_pdb_paths(pdb_paths, pdb_dir, pdb_pattern, recursive)
        self.tgt_chains = self._normalize_chain_specs(tgt_chains, len(self.pdb_paths), 'tgt_chains')
        self.lig_chains = self._normalize_chain_specs(lig_chains, len(self.pdb_paths), 'lig_chains')
        self.ligand_size_mode = ligand_size_mode
        self.cplxs, self.pocket_block_ids, self.ref_ligand_sizes = [], [], []
        preserve_residue_names = getattr(self.config, 'preserve_residue_names', None)
        for path, tgt, lig in zip(self.pdb_paths, self.tgt_chains, self.lig_chains):
            cplx, pocket_block_ids = self._process_pdb(path, tgt, lig, preserve_residue_names=preserve_residue_names)
            self.cplxs.append(cplx)
            self.pocket_block_ids.append(pocket_block_ids)
            self.ref_ligand_sizes.append(self._ligand_size(cplx, lig[0], ligand_size_mode))
        self.n_samples = n_samples

    def _resolve_pdb_paths(self, pdb_paths, pdb_dir, pdb_pattern, recursive):
        if pdb_paths is not None and pdb_dir is not None:
            raise ValueError('Specify either dataset.pdb_paths or dataset.pdb_dir, not both')
        if pdb_dir is not None:
            root = Path(pdb_dir)
            iterator = root.rglob(pdb_pattern) if recursive else root.glob(pdb_pattern)
            pdb_paths = sorted(str(path) for path in iterator if path.is_file())
            if not pdb_paths:
                raise ValueError(f'No PDB files matching {pdb_pattern!r} found in {pdb_dir}')
        if pdb_paths is None:
            raise ValueError('Set either dataset.pdb_paths or dataset.pdb_dir')
        return [str(path) for path in pdb_paths]

    def _as_chain_list(self, value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError(f'Chain specification must be a string or list of strings, got {type(value)}')

    def _normalize_chain_specs(self, value, n_items, name):
        if value is None:
            raise ValueError(f'Set dataset.{name}')
        if isinstance(value, str):
            return [[value] for _ in range(n_items)]
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            raise ValueError(f'dataset.{name} must be a string, list of strings, or list of chain lists')

        has_nested = any(isinstance(item, (list, tuple)) for item in value)
        if has_nested:
            specs = [self._as_chain_list(item) for item in value]
            if len(specs) == 1 and n_items > 1:
                return specs * n_items
            if len(specs) != n_items:
                raise ValueError(f'dataset.{name} has {len(specs)} entries for {n_items} PDB files')
            return specs

        if n_items == 1:
            return [self._as_chain_list(value)]
        if len(value) == 1:
            return [[str(value[0])] for _ in range(n_items)]
        if len(value) == n_items:
            return [[str(item)] for item in value]
        raise ValueError(
            f'dataset.{name} is ambiguous for {n_items} PDB files. '
            f'Use a string for one global chain, a single nested list like [[A, B]] for global '
            f'multiple chains, or a list of {n_items} per-file entries.'
        )

    def _has_atom(self, block, atom_name):
        return any(atom.name == atom_name for atom in block)

    def _ligand_size(self, cplx, lig_chain, mode):
        blocks = list(cplx[lig_chain])
        mode = str(mode)
        if mode == 'blocks':
            return len(blocks)
        if mode in {'ca', 'res_ca', 'residue_ca'}:
            return len({
                tuple(block.id)
                for block in blocks
                if block.name not in {'ACE', 'NME', 'CAP'} and self._has_atom(block, 'CA')
            })
        if mode == 'backbone':
            return sum(
                1 for block in blocks
                if all(self._has_atom(block, atom_name) for atom_name in ('N', 'CA', 'C', 'O'))
            )
        raise ValueError(f'Unknown ligand_size_mode: {mode}')

    def __getitem__(self, idx):
        idx = idx // self.n_samples
        cplx, pocket_block_ids = self.cplxs[idx], self.pocket_block_ids[idx]
        lig_chain = self.lig_chains[idx][0]
        lig_term_rel_pos = compute_terminal_rel_pos(list(cplx[lig_chain]))

        # edit the cplx with template (create new cplx of ligand and merge)
        cplx_desc = self.config(ComplexDesc(
            id=os.path.basename(os.path.splitext(self.pdb_paths[idx])[0]),
            cplx=cplx,
            tgt_chains=list(self.tgt_chains[idx]),
            lig_chains=list(self.lig_chains[idx]),
            pocket_block_ids=pocket_block_ids,
            ref_ligand_size=self.ref_ligand_sizes[idx],
        ))

        dummy_lig_block_ids = cplx_desc.lig_block_ids

        data = transform_data(cplx_desc.cplx, pocket_block_ids + dummy_lig_block_ids)
        data['generate_mask'] = torch.tensor(cplx_desc.generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(cplx_desc.center_mask, dtype=torch.bool)
        term_rel_pos = np.zeros((int(data['lengths']), 15), dtype=np.float32)
        if len(dummy_lig_block_ids):
            term_rel_pos[-len(dummy_lig_block_ids):] = lig_term_rel_pos
        data['term_rel_pos'] = torch.tensor(term_rel_pos, dtype=torch.float)
        if cplx_desc.topo_generate_mask is not None:
            data['topo_generate_mask'] = torch.tensor(cplx_desc.topo_generate_mask, dtype=torch.bool)

        data = self.config.set_control(data, cplx_desc=cplx_desc)
        data['cplx_desc'] = cplx_desc

        return data

    def __len__(self):
        return len(self.cplxs) * self.n_samples
    
    def _process_pdb(self, pdb_path, tgt_chains, lig_chains, pocket_dist_th=10.0, preserve_residue_names=None):
        kwargs = {}
        if preserve_residue_names:
            if not _pdb_to_complex_supports_preserve_residue_names():
                raise TypeError(
                    'This pdb_to_complex() does not support preserve_residue_names; '
                    'update data/bioparse/parser/pdb_to_complex.py or use a template '
                    'that does not require preserved ligand residues.'
                )
            kwargs['preserve_residue_names'] = preserve_residue_names
        cplx = pdb_to_complex(pdb_path, tgt_chains + lig_chains, **kwargs)
        frag_lig_chains = []
        for mol in cplx:
            for c in lig_chains:
                if c + '_' in mol.id: frag_lig_chains.append(mol.id) # small molecules fragmented
        pocket_block_ids, _ = compute_pocket(cplx, tgt_chains, list(lig_chains) + frag_lig_chains, dist_th=pocket_dist_th)
        return cplx, pocket_block_ids
    
    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'bonds': # need to add offsets
                offset = 0
                for i, bonds in enumerate(values):
                    bonds[:, :2] = bonds[:, :2] + offset # src/dst
                    offset += len(batch[i]['A'])
                results[key] = torch.cat(values, dim=0)
            elif key == 'cplx_desc':
                results[key] = values
            else:
                results[key] = torch.cat(values, dim=0)
        return results
    

if __name__ == '__main__':
    import sys
    dataset = PDBDataset([sys.argv[1]], sys.argv[2], sys.argv[3], BaseTemplate(10, 12)) # e.g. xxx.pdb AB CD
    print(dataset[0])
