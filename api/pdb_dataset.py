#!/usr/bin/python
# -*- coding:utf-8 -*-
import csv
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
            pdb_path: str = None,
            pdb_paths: List[str] = None,
            tgt_chains: List[List[str]] = None,
            lig_chains: List[List[str]] = None,
            template_config: BaseTemplate = None,
            n_samples: int = 1,
            pdb_dir: str = None,
            pdb_pattern: str = '*.pdb',
            recursive: bool = False,
            ligand_size_mode: str = 'ca',
            generated_lig_chain: str = 'L',
            pocket_threshold: float = 10.0,
            pocket_residues=None,
            pocket_residue_tsv: str = None,
            pocket_residues_tsv: str = None,
        ):
        super().__init__()
        self.config = template_config
        self._validate_ligand_size_mode(ligand_size_mode)
        self.ligand_size_mode = 'ca'
        self.pocket_threshold = float(pocket_threshold)
        self.pdb_paths = self._resolve_pdb_paths(pdb_path, pdb_paths, pdb_dir, pdb_pattern, recursive)
        self.pocket_residues_by_file = self._resolve_pocket_residues(
            pocket_residues,
            pocket_residue_tsv,
            pocket_residues_tsv,
        )
        self.tgt_chains = self._normalize_chain_specs(tgt_chains, len(self.pdb_paths), 'tgt_chains')
        self.target_only = self._is_empty_chain_spec(lig_chains)
        if self.target_only:
            if getattr(self.config, 'size_from_ligand', False):
                raise ValueError(
                    'dataset.lig_chains is omitted, so size_from_ligand=True cannot be used. '
                    'Set template size_min/size_max or provide a reference ligand chain.'
                )
            self.lig_chains = self._normalize_chain_specs(
                generated_lig_chain,
                len(self.pdb_paths),
                'generated_lig_chain',
            )
            for tgt, lig in zip(self.tgt_chains, self.lig_chains):
                overlap = set(tgt) & set(lig)
                if overlap:
                    raise ValueError(
                        'dataset.generated_lig_chain must not overlap dataset.tgt_chains '
                        f'in target-only mode: {sorted(overlap)}'
                    )
        else:
            self.lig_chains = self._normalize_chain_specs(lig_chains, len(self.pdb_paths), 'lig_chains')
        self.cplxs, self.pocket_block_ids, self.ref_ligand_sizes = [], [], []
        preserve_residue_names = getattr(self.config, 'preserve_residue_names', None)
        for path, tgt, lig in zip(self.pdb_paths, self.tgt_chains, self.lig_chains):
            pocket_residues_for_path = self._pocket_residues_for_path(path)
            cplx, pocket_block_ids = self._process_pdb(
                path,
                tgt,
                lig,
                has_reference_ligand=not self.target_only,
                pocket_dist_th=self.pocket_threshold,
                pocket_residues=pocket_residues_for_path,
                preserve_residue_names=preserve_residue_names,
            )
            self.cplxs.append(cplx)
            self.pocket_block_ids.append(pocket_block_ids)
            self.ref_ligand_sizes.append(
                None if self.target_only else self._ligand_size(cplx, lig[0])
            )
        self.n_samples = n_samples

    def _validate_ligand_size_mode(self, ligand_size_mode):
        if str(ligand_size_mode).strip().lower() != 'ca':
            raise ValueError('dataset.ligand_size_mode is fixed to "ca" and cannot be modified')

    def _resolve_pdb_paths(self, pdb_path, pdb_paths, pdb_dir, pdb_pattern, recursive):
        sources = [pdb_path is not None, pdb_paths is not None, pdb_dir is not None]
        if sum(sources) > 1:
            raise ValueError('Specify only one of dataset.pdb_path, dataset.pdb_paths, or dataset.pdb_dir')
        if pdb_path is not None:
            path = Path(pdb_path)
            if not path.is_file():
                raise ValueError(f'dataset.pdb_path does not exist or is not a file: {pdb_path}')
            return [str(path)]
        if pdb_dir is not None:
            root = Path(pdb_dir)
            iterator = root.rglob(pdb_pattern) if recursive else root.glob(pdb_pattern)
            pdb_paths = sorted(str(path) for path in iterator if path.is_file())
            if not pdb_paths:
                raise ValueError(f'No PDB files matching {pdb_pattern!r} found in {pdb_dir}')
        if pdb_paths is None:
            raise ValueError('Set one of dataset.pdb_path, dataset.pdb_paths, or dataset.pdb_dir')
        if isinstance(pdb_paths, str):
            pdb_paths = [pdb_paths]
        return [str(path) for path in pdb_paths]

    def _normalize_ins_code(self, value):
        if value is None:
            return ''
        value = str(value).strip()
        if value in {'.', '?'}:
            return ''
        return value

    def _normalize_pocket_residue(self, residue):
        if isinstance(residue, dict):
            chain = self._get_first(residue, ('chain_id', 'chainID', 'chain'))
            res_id = self._get_first(residue, ('res_id', 'resID', 'residue_id', 'residue_number', 'resseq'))
            ins_code = self._get_first(
                residue,
                ('res_ins_code', 'ins_code', 'insertion_code', 'pdbx_PDB_ins_code'),
                '',
            )
        elif isinstance(residue, (list, tuple)) and len(residue) in {2, 3}:
            chain, res_id = residue[:2]
            ins_code = residue[2] if len(residue) == 3 else ''
        else:
            raise ValueError(
                'Pocket residues must be [chain_id, res_id, res_ins_code] tuples '
                'or dictionaries with chain/residue fields'
            )
        if chain is None or res_id is None:
            raise ValueError(f'Pocket residue is missing chain_id or res_id: {residue}')
        return (str(chain), (int(res_id), self._normalize_ins_code(ins_code)))

    def _normalize_pocket_residue_list(self, residues):
        if residues is None:
            return None
        if isinstance(residues, dict) and self._looks_like_residue_dict(residues):
            residues = [residues]
        if not isinstance(residues, (list, tuple)):
            raise ValueError('dataset.pocket_residues must be a residue list or a mapping from PDB names to residue lists')
        return [self._normalize_pocket_residue(residue) for residue in residues]

    def _looks_like_residue_dict(self, value):
        if not isinstance(value, dict):
            return False
        keys = set(value)
        return bool(keys & {'chain_id', 'chainID', 'chain'}) and bool(
            keys & {'res_id', 'resID', 'residue_id', 'residue_number', 'resseq'}
        )

    def _get_first(self, mapping, names, default=None):
        for name in names:
            if name in mapping:
                return mapping[name]
        return default

    def _resolve_pocket_residues(self, pocket_residues, pocket_residue_tsv, pocket_residues_tsv):
        tsv_paths = [path for path in (pocket_residue_tsv, pocket_residues_tsv) if path is not None]
        if len(tsv_paths) > 1:
            raise ValueError('Specify only one of dataset.pocket_residue_tsv or dataset.pocket_residues_tsv')
        if pocket_residues is not None and tsv_paths:
            raise ValueError('Specify either dataset.pocket_residues or dataset.pocket_residues_tsv, not both')
        if tsv_paths:
            return self._load_pocket_residue_tsv(tsv_paths[0])
        if pocket_residues is None:
            return None
        if isinstance(pocket_residues, dict) and not self._looks_like_residue_dict(pocket_residues):
            return {
                str(key): self._normalize_pocket_residue_list(value)
                for key, value in pocket_residues.items()
            }
        return {'*': self._normalize_pocket_residue_list(pocket_residues)}

    def _load_pocket_residue_tsv(self, tsv_path):
        path = Path(tsv_path)
        if not path.is_file():
            raise ValueError(f'dataset.pocket_residues_tsv does not exist or is not a file: {tsv_path}')
        by_file = {}
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if reader.fieldnames is None:
                raise ValueError(f'Pocket residue TSV is missing a header: {tsv_path}')
            for row in reader:
                pdb_name = self._get_first(row, ('pdb_file', 'pdb_path', 'filename', 'file', 'pdb'))
                if not pdb_name:
                    raise ValueError(
                        'Pocket residue TSV must include a pdb_file, pdb_path, filename, file, or pdb column'
                    )
                by_file.setdefault(str(pdb_name), []).append(self._normalize_pocket_residue(row))
        return by_file

    def _pdb_match_keys(self, path):
        path = Path(path)
        keys = {str(path), path.name, path.stem}
        try:
            keys.add(str(path.resolve()))
        except OSError:
            pass
        return keys

    def _pocket_residues_for_path(self, path):
        if self.pocket_residues_by_file is None:
            return None
        if '*' in self.pocket_residues_by_file:
            return self.pocket_residues_by_file['*']
        keys = self._pdb_match_keys(path)
        for key, residues in self.pocket_residues_by_file.items():
            key_path = Path(str(key))
            if str(key) in keys or key_path.name in keys or key_path.stem in keys:
                return residues
        raise ValueError(
            f'No manual pocket residues found for {path}. '
            f'Available pocket residue keys: {sorted(self.pocket_residues_by_file)}'
        )

    def _as_chain_list(self, value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError(f'Chain specification must be a string or list of strings, got {type(value)}')

    def _is_empty_chain_spec(self, value):
        return value is None or value == '' or value == []

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

    def _ligand_size(self, cplx, lig_chain):
        blocks = list(cplx[lig_chain])
        return len({
            tuple(block.id)
            for block in blocks
            if block.name not in {'ACE', 'NME', 'CAP'} and self._has_atom(block, 'CA')
        })

    def __getitem__(self, idx):
        idx = idx // self.n_samples
        cplx, pocket_block_ids = self.cplxs[idx], self.pocket_block_ids[idx]
        lig_chain = self.lig_chains[idx][0]
        if self.target_only:
            lig_term_rel_pos = np.zeros(15, dtype=np.float32)
        else:
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

    def _target_block_ids(self, cplx, tgt_chains):
        return [
            (chain_id, block.id)
            for chain_id in tgt_chains
            for block in cplx[chain_id]
        ]

    def _manual_pocket_block_ids(self, cplx, tgt_chains, pocket_residues):
        tgt_chain_set = set(tgt_chains)
        available = set(self._target_block_ids(cplx, tgt_chains))
        selected = []
        missing = []
        seen = set()
        for chain_id, block_id in pocket_residues:
            block_ref = (chain_id, block_id)
            if chain_id not in tgt_chain_set or block_ref not in available:
                missing.append(block_ref)
                continue
            if block_ref not in seen:
                selected.append(block_ref)
                seen.add(block_ref)
        if missing:
            raise ValueError(f'Manual pocket residues are not present in target chains: {missing}')
        if not selected:
            raise ValueError('Manual pocket residue selection is empty')
        return selected

    def _process_pdb(
            self,
            pdb_path,
            tgt_chains,
            lig_chains,
            pocket_dist_th=10.0,
            has_reference_ligand=True,
            pocket_residues=None,
            preserve_residue_names=None,
        ):
        kwargs = {}
        if preserve_residue_names:
            if not _pdb_to_complex_supports_preserve_residue_names():
                raise TypeError(
                    'This pdb_to_complex() does not support preserve_residue_names; '
                    'update data/bioparse/parser/pdb_to_complex.py or use a template '
                    'that does not require preserved ligand residues.'
                )
            kwargs['preserve_residue_names'] = preserve_residue_names
        selected_chains = tgt_chains + (lig_chains if has_reference_ligand else [])
        cplx = pdb_to_complex(pdb_path, selected_chains, **kwargs)
        if pocket_residues is not None:
            return cplx, self._manual_pocket_block_ids(cplx, tgt_chains, pocket_residues)
        if not has_reference_ligand:
            return cplx, self._target_block_ids(cplx, tgt_chains)

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
