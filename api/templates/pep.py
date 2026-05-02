#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import utils.register as R
from data.bioparse.hierarchy import Atom, Block, Bond, BondType, Complex, Molecule, remove_mols
from data.peptide import compute_terminal_rel_pos_from_sdf

from .base import BaseTemplate, ComplexDesc


class TermRelPosSDFSampler:

    def __init__(self, config):
        self.config = dict(config)
        sdf_path = self.config.get('path', self.config.get('sdf_path', None))
        if sdf_path is None:
            raise ValueError('term_rel_pos_sdf requires "path" or "sdf_path"')

        self.paths = self._resolve_paths(sdf_path)
        self.n_term_atom_indices, self.c_term_atom_indices = self._resolve_atom_indices()
        self.index_base = self.config.get('index_base', 0)
        self.mol_index = self.config.get('mol_index', 0)
        self._cache = {}
        self._path_order = []
        self._path_cursor = 0
        self._last_path = None

    def _resolve_paths(self, sdf_path):
        path = Path(sdf_path)
        if path.is_dir():
            pattern = self.config.get('pattern', '*.sdf')
            iterator = path.rglob(pattern) if self.config.get('recursive', False) else path.glob(pattern)
            paths = sorted(str(candidate) for candidate in iterator if candidate.is_file())
            if not paths:
                raise ValueError(f'No SDF files matching {pattern!r} found in directory: {sdf_path}')
            return paths
        if not path.is_file():
            raise ValueError(f'term_rel_pos_sdf path does not exist: {sdf_path}')
        return [str(path)]

    def _resolve_atom_indices(self):
        n_term_atom_indices = self.config.get('n_term_atom_indices', None)
        c_term_atom_indices = self.config.get('c_term_atom_indices', None)
        if n_term_atom_indices is not None and c_term_atom_indices is not None:
            return list(n_term_atom_indices), list(c_term_atom_indices)

        atom_indices = self.config.get('atom_indices', None)
        if atom_indices is None:
            raise ValueError(
                'term_rel_pos_sdf requires either n_term_atom_indices/c_term_atom_indices '
                'or atom_indices'
            )
        if len(atom_indices) == 10:
            return list(atom_indices[:5]), list(atom_indices[5:])
        if len(atom_indices) == 2 and len(atom_indices[0]) == 5 and len(atom_indices[1]) == 5:
            return list(atom_indices[0]), list(atom_indices[1])
        raise ValueError('atom_indices must be either a 10-item list or a pair of 5-item lists')

    def _term_rel_pos(self, sdf_path):
        if sdf_path not in self._cache:
            self._cache[sdf_path] = compute_terminal_rel_pos_from_sdf(
                sdf_path=sdf_path,
                n_term_atom_indices=self.n_term_atom_indices,
                c_term_atom_indices=self.c_term_atom_indices,
                index_base=self.index_base,
                mol_index=self.mol_index,
            )
        return self._cache[sdf_path]

    def _next_path(self):
        if len(self.paths) == 1:
            return self.paths[0]

        if self._path_cursor >= len(self._path_order):
            self._path_order = list(self.paths)
            np.random.shuffle(self._path_order)
            if self._last_path is not None and self._path_order[0] == self._last_path:
                self._path_order[0], self._path_order[1] = self._path_order[1], self._path_order[0]
            self._path_cursor = 0

        sdf_path = self._path_order[self._path_cursor]
        self._path_cursor += 1
        self._last_path = sdf_path
        return sdf_path

    def sample(self):
        sdf_path = str(self._next_path())
        term_rel_pos = self._term_rel_pos(sdf_path)
        metadata = {
            'term_rel_pos_sdf_path': sdf_path,
            'term_rel_pos_atom_indices': list(self.n_term_atom_indices) + list(self.c_term_atom_indices),
            'term_rel_pos_index_base': self.index_base,
            'term_rel_pos_mol_index': self.mol_index,
        }
        for key in (
                'gen_atom_indices',
                'gen_index_base',
                'ref_ca_distance_slots',
                'gen_ca_distance_slots',
            ):
            if key in self.config:
                metadata[key] = self.config[key]
        return term_rel_pos, metadata


@R.register('LinearPeptide')
class LinearPeptide(BaseTemplate):

    def __init__(
            self,
            size_min=None,
            size_max=None,
            term_rel_pos=None,
            term_rel_pos_sdf=None,
            size_from_ligand=False,
            ligand_size_offset=0,
        ):
        super().__init__(
            size_min,
            size_max,
            size_from_ligand=size_from_ligand,
            ligand_size_offset=ligand_size_offset,
        )
        if term_rel_pos is not None and term_rel_pos_sdf is not None:
            raise ValueError('Specify either term_rel_pos or term_rel_pos_sdf, not both')
        self.term_rel_pos_sampler = None
        if term_rel_pos_sdf is not None:
            self.term_rel_pos = None
            self.term_rel_pos_sampler = TermRelPosSDFSampler(term_rel_pos_sdf)
        elif term_rel_pos is None:
            self.term_rel_pos = None
        else:
            self.term_rel_pos = np.asarray(term_rel_pos, dtype=np.float32)
            if self.term_rel_pos.shape != (15,):
                raise ValueError('term_rel_pos must be a 15-dimensional CFG condition')

    def dummy_lig_block_bonds(self, cplx_desc, size):
        blocks = [Block(
            name='GLY', # use glycine instead of UNK so that is_aa can be identified as True
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 1, '')
        ) for i in range(size)]
        bonds = []
        return blocks, bonds

    def set_control(self, data, cplx_desc=None):
        term_rel_pos = self.term_rel_pos
        control = None
        if self.term_rel_pos_sampler is not None:
            term_rel_pos, control = self.term_rel_pos_sampler.sample()
        if term_rel_pos is not None:
            term_rel_pos = torch.as_tensor(
                term_rel_pos,
                dtype=data['term_rel_pos'].dtype,
                device=data['term_rel_pos'].device
            )
            data['term_rel_pos'].zero_()
            data['term_rel_pos'][data['generate_mask']] = term_rel_pos
        if cplx_desc is not None and control is not None:
            cplx_desc.control = control
        return data


@R.register('CappedLinearPeptide')
class CappedLinearPeptide(LinearPeptide):
    """Generate peptide residues while preserving reference terminal cap blocks."""

    def __init__(
            self,
            size_min=None,
            size_max=None,
            term_rel_pos=None,
            term_rel_pos_sdf=None,
            size_from_ligand=False,
            ligand_size_offset=0,
            n_cap_block_index=0,
            c_cap_block_index=-1,
            n_cap_residue_name=None,
            c_cap_residue_name=None,
            preserve_residue_names=None,
        ):
        super().__init__(
            size_min,
            size_max,
            term_rel_pos,
            term_rel_pos_sdf,
            size_from_ligand=size_from_ligand,
            ligand_size_offset=ligand_size_offset,
        )
        self.n_cap_block_index = n_cap_block_index
        self.c_cap_block_index = c_cap_block_index
        self.n_cap_residue_name = n_cap_residue_name
        self.c_cap_residue_name = c_cap_residue_name
        if preserve_residue_names is None:
            preserve_residue_names = ('ACE', 'NME', 'LI0', 'LI1')
        self.preserve_residue_names = tuple(str(name) for name in preserve_residue_names)

    def _copy_block(self, block, new_id):
        block = deepcopy(block)
        block.id = new_id
        return block

    def _copy_cap_bonds(self, source_cplx, source_chain, source_block, target_mol_idx, target_block_idx):
        source_mol_idx = source_cplx.id2idx[source_chain]
        source_block_idx = source_cplx[source_chain].id2idx[source_block.id]
        bonds = []
        for bond in source_cplx.bonds:
            if bond.index1[:2] != (source_mol_idx, source_block_idx):
                continue
            if bond.index2[:2] != (source_mol_idx, source_block_idx):
                continue
            bonds.append(Bond(
                index1=(target_mol_idx, target_block_idx, bond.index1[2]),
                index2=(target_mol_idx, target_block_idx, bond.index2[2]),
                bond_type=bond.bond_type
            ))
        return bonds

    def _validate_cap_block(self, block, label):
        original_name = block.properties.get('original_name', None)
        if original_name is not None and original_name != block.name:
            raise ValueError(
                f'{label} cap block {block.id} is a tokenizer fragment "{block.name}" '
                f'from residue "{original_name}". Add "{original_name}" to '
                'preserve_residue_names or choose the cap residue after preserving it.'
            )

    def _select_cap_block(self, ref_blocks, residue_name, block_index, label):
        if residue_name is None:
            return ref_blocks[block_index]

        residue_name = str(residue_name)
        matches = [
            block for block in ref_blocks
            if block.name == residue_name or block.properties.get('original_name', None) == residue_name
        ]
        if len(matches) != 1:
            available = [
                block.properties.get('original_name', block.name)
                for block in ref_blocks
            ]
            raise ValueError(
                f'{label} cap residue "{residue_name}" matched {len(matches)} blocks. '
                f'Available ligand blocks: {available}'
            )
        return matches[0]

    def __call__(self, cplx_desc: ComplexDesc) -> ComplexDesc:
        cplx_desc = deepcopy(cplx_desc)
        lig_chain = cplx_desc.lig_chains[0]
        ref_blocks = list(cplx_desc.cplx[lig_chain])
        if len(ref_blocks) < 2:
            raise ValueError('CappedLinearPeptide requires a reference ligand chain with at least two cap blocks')

        n_cap_ref = self._select_cap_block(
            ref_blocks, self.n_cap_residue_name, self.n_cap_block_index, 'N-terminal'
        )
        c_cap_ref = self._select_cap_block(
            ref_blocks, self.c_cap_residue_name, self.c_cap_block_index, 'C-terminal'
        )
        self._validate_cap_block(n_cap_ref, 'N-terminal')
        self._validate_cap_block(c_cap_ref, 'C-terminal')
        core_size = self.sample_size(cplx_desc)

        blocks = [self._copy_block(n_cap_ref, (1, ''))]
        blocks.extend(Block(
            name='GLY',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 2, '')
        ) for i in range(core_size))
        blocks.append(self._copy_block(c_cap_ref, (core_size + 2, '')))

        cplx_without_ligand = remove_mols(cplx_desc.cplx, cplx_desc.lig_chains)
        target_mol_idx = len(cplx_without_ligand.molecules)
        cap_bonds = []
        cap_bonds.extend(self._copy_cap_bonds(cplx_desc.cplx, lig_chain, n_cap_ref, target_mol_idx, 0))
        cap_bonds.extend(self._copy_cap_bonds(cplx_desc.cplx, lig_chain, c_cap_ref, target_mol_idx, len(blocks) - 1))

        dummy_mol = Molecule(
            name='dummy',
            blocks=blocks,
            id=lig_chain,
        )
        cplx_desc.cplx = Complex(
            name=cplx_without_ligand.name,
            molecules=cplx_without_ligand.molecules + [dummy_mol],
            bonds=cplx_without_ligand.bonds + cap_bonds,
            properties=cplx_without_ligand.properties
        )
        cplx_desc = self.set_desc_attr(cplx_desc)
        n_pocket = len(cplx_desc.pocket_block_ids)
        n_lig = len(cplx_desc.lig_block_ids)
        cplx_desc.topo_generate_mask = (
            [0 for _ in range(n_pocket)]
            + [0]
            + [1 for _ in range(n_lig - 2)]
            + [0]
        )
        return cplx_desc


@R.register('DeNovoCappedLinearPeptide')
class DeNovoCappedLinearPeptide(LinearPeptide):
    """Generate a peptide with synthetic ACE/NME cap blocks."""

    def __init__(
            self,
            size_min=None,
            size_max=None,
            term_rel_pos=None,
            term_rel_pos_sdf=None,
            size_from_ligand=False,
            ligand_size_offset=0,
            n_cap_name='ACE',
            c_cap_name='NME',
        ):
        super().__init__(
            size_min,
            size_max,
            term_rel_pos,
            term_rel_pos_sdf,
            size_from_ligand=size_from_ligand,
            ligand_size_offset=ligand_size_offset,
        )
        self.n_cap_name = str(n_cap_name)
        self.c_cap_name = str(c_cap_name)

    def _ace_block(self, block_id):
        return Block(
            name=self.n_cap_name,
            atoms=[
                Atom(name='CA', coordinate=[0, 0, 0], element='C', id='CA'),
                Atom(name='C', coordinate=[0, 0, 0], element='C', id='C'),
                Atom(name='O', coordinate=[0, 0, 0], element='O', id='O'),
            ],
            id=block_id,
            properties={'original_name': self.n_cap_name}
        )

    def _nme_block(self, block_id):
        return Block(
            name=self.c_cap_name,
            atoms=[
                Atom(name='N', coordinate=[0, 0, 0], element='N', id='N'),
                Atom(name='CA', coordinate=[0, 0, 0], element='C', id='CA'),
            ],
            id=block_id,
            properties={'original_name': self.c_cap_name}
        )

    def _cap_bonds(self, target_mol_idx, n_cap_block_idx, c_cap_block_idx):
        return [
            Bond((target_mol_idx, n_cap_block_idx, 0), (target_mol_idx, n_cap_block_idx, 1), BondType.SINGLE),
            Bond((target_mol_idx, n_cap_block_idx, 1), (target_mol_idx, n_cap_block_idx, 2), BondType.DOUBLE),
            Bond((target_mol_idx, c_cap_block_idx, 0), (target_mol_idx, c_cap_block_idx, 1), BondType.SINGLE),
        ]

    def set_control(self, data, cplx_desc=None):
        data['term_rel_pos'].zero_()
        return super().set_control(data, cplx_desc=cplx_desc)

    def __call__(self, cplx_desc: ComplexDesc) -> ComplexDesc:
        cplx_desc = deepcopy(cplx_desc)
        lig_chain = cplx_desc.lig_chains[0]
        core_size = self.sample_size(cplx_desc)

        blocks = [self._ace_block((1, ''))]
        blocks.extend(Block(
            name='GLY',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 2, '')
        ) for i in range(core_size))
        blocks.append(self._nme_block((core_size + 2, '')))

        cplx_without_ligand = remove_mols(cplx_desc.cplx, cplx_desc.lig_chains)
        target_mol_idx = len(cplx_without_ligand.molecules)
        dummy_mol = Molecule(
            name='dummy',
            blocks=blocks,
            id=lig_chain,
        )
        cplx_desc.cplx = Complex(
            name=cplx_without_ligand.name,
            molecules=cplx_without_ligand.molecules + [dummy_mol],
            bonds=cplx_without_ligand.bonds + self._cap_bonds(target_mol_idx, 0, len(blocks) - 1),
            properties=cplx_without_ligand.properties
        )
        cplx_desc = self.set_desc_attr(cplx_desc)
        n_pocket = len(cplx_desc.pocket_block_ids)
        n_lig = len(cplx_desc.lig_block_ids)
        cplx_desc.topo_generate_mask = (
            [0 for _ in range(n_pocket)]
            + [0]
            + [1 for _ in range(n_lig - 2)]
            + [0]
        )
        return cplx_desc
