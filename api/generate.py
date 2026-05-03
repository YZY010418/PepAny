#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil
import argparse
import csv
import json
import re
import random
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
try:
    import fcntl
except ImportError:
    fcntl = None
from copy import deepcopy
from pathlib import Path

import yaml
import torch
import numpy as np
from rdkit import Chem

from data.bioparse.hierarchy import Atom, Block, Bond, BondType, Complex, Molecule
from data.bioparse.parser.cif_to_complex import cif_to_complex
from data.bioparse.utils import bond_type_from_rdkit
from data.bioparse.writer.complex_to_cif import complex_to_cif
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from evaluation.terminal_amide_rmsd import (
    align_ref_to_generated,
    detect_clashes,
    detect_receptor_clashes,
    get_positions,
    heavy_atom_mol_with_map,
    infer_generated_indexes,
    kabsch_align,
    load_sdf_mol,
    merge_terminal_cap,
    remap_indexes,
    remove_atoms_with_map,
    rmsd,
    write_sdf,
)
from utils.config_utils import overwrite_values
from utils.io import torch_load_full
from utils.logger import print_log
from utils.random_seed import setup_seed
import utils.register as R
from models.LDM.data_utils import OverwriteTask, _get_item, Recorder

from .pdb_dataset import PDBDataset, ComplexDesc
from .templates import BaseTemplate


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/model.ckpt', help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated peptides')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--seed', type=int, default=None, help='Random seed, overrides config seed')
    parser.add_argument('--num_shards', '--total_gpus', dest='num_shards', type=int, default=None, help='Total number of generation shards')
    parser.add_argument('--shard_id', '--worker_id', dest='shard_id', type=int, default=None, help='Index of this generation shard, from 0 to num_shards - 1')
    parser.add_argument('--num_workers', type=int, default=None, help='CPU workers for generation batch prefetch, overrides config num_workers')
    parser.add_argument('--postprocess_workers', type=int, default=None, help='CPU workers for candidate post-processing, overrides config generation_postprocess.num_workers')
    parser.add_argument('--n_cpus', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    return parser.parse_known_args()


# utils
def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


_PREFETCH_DATASET = None


def _init_batch_prefetch_worker(dataset, torch_num_threads):
    global _PREFETCH_DATASET
    _PREFETCH_DATASET = dataset
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))


def _setup_prefetch_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))


def _collate_batch_worker(task):
    batch_list, task_seed = task
    if task_seed is not None:
        _setup_prefetch_seed(task_seed)
    return batch_list, _PREFETCH_DATASET.collate_fn([
        _PREFETCH_DATASET[i]
        for i, _ in batch_list
    ])


def _init_postprocess_worker(torch_num_threads):
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))


class BatchPrefetcher:

    def __init__(
            self,
            dataset,
            recorder,
            batch_size,
            num_workers=0,
            prefetch_batches=2,
            backend='process',
            start_method='spawn',
            seed=None,
            torch_num_threads=1,
        ):
        self.dataset = dataset
        self.recorder = recorder
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.backend = str(backend).lower()
        self.start_method = str(start_method)
        self.seed = seed
        self.torch_num_threads = torch_num_threads
        self.executor = None
        self.pending = deque()
        self.task_index = 0

    def __enter__(self):
        if self.num_workers > 0:
            if self.backend == 'thread':
                self.executor = ThreadPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_init_batch_prefetch_worker,
                    initargs=(self.dataset, None),
                )
            elif self.backend == 'process':
                mp_context = mp.get_context(self.start_method)
                self.executor = ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_init_batch_prefetch_worker,
                    initargs=(self.dataset, self.torch_num_threads),
                    mp_context=mp_context,
                )
            else:
                raise ValueError('generation prefetch backend must be "process" or "thread"')
            self._fill()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._shutdown_executor()

    def _shutdown_executor(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def _next_seed(self):
        if self.seed is None:
            return None
        return int(self.seed) + self.task_index * 1000003

    def _prepare_serial(self, batch_list, task_seed=None):
        if task_seed is not None:
            setup_seed(int(task_seed))
        return batch_list, self.dataset.collate_fn([
            self.dataset[i]
            for i, _ in batch_list
        ])

    def _submit_next(self):
        if self.recorder.is_finished():
            return False
        batch_list = self.recorder.get_next_batch_list(self.batch_size)
        if not batch_list:
            return False
        task_seed = self._next_seed() if self.backend == 'process' and self.executor is not None else None
        self.task_index += 1
        if self.executor is None:
            self.pending.append(('ready', batch_list, self._prepare_serial(batch_list, task_seed)))
        else:
            try:
                self.pending.append((
                    'future',
                    batch_list,
                    self.executor.submit(
                        _collate_batch_worker,
                        (batch_list, task_seed),
                    ),
                ))
            except Exception as exc:
                print_log(
                    f'Generation prefetch worker submission failed; '
                    f'falling back to serial batch preparation: {exc}',
                    level='WARN',
                )
                self._shutdown_executor()
                self.pending.append(('ready', batch_list, self._prepare_serial(batch_list)))
        return True

    def _fill(self):
        while len(self.pending) < self.prefetch_batches:
            if not self._submit_next():
                break

    def next(self):
        if self.executor is None:
            if not self.pending:
                self._submit_next()
            if not self.pending:
                raise StopIteration
            kind, batch_list, payload = self.pending.popleft()
            if kind == 'ready':
                return payload
            return self._prepare_serial(batch_list)

        self._fill()
        if not self.pending:
            raise StopIteration
        kind, batch_list, payload = self.pending.popleft()
        if kind == 'ready':
            return payload
        future = payload
        try:
            result = future.result()
        except Exception as exc:
            print_log(
                f'Generation batch prefetch failed; falling back to serial batch '
                f'preparation. Original error: {exc}',
                level='WARN',
            )
            self._shutdown_executor()
            result = self._prepare_serial(batch_list)
        if self.executor is not None:
            self._fill()
        return result


def data_to_cplx(cplx_desc: ComplexDesc, template: BaseTemplate, S: list, X: list, A: list, ll: list, inter_bonds: tuple, intra_bonds: tuple, out_path: str, check_validity: bool=False, check_filters: bool=False):
    '''
        Args:
            bonds: [row, col, prob, type], row and col are atom index, prob has confidence and distance
    '''

    task = OverwriteTask(
        cplx = cplx_desc.cplx,
        select_indexes = cplx_desc.pocket_block_ids + cplx_desc.lig_block_ids,
        generate_mask = cplx_desc.generate_mask,
        target_chain_ids = cplx_desc.tgt_chains,
        ligand_chain_ids = cplx_desc.lig_chains,
        S = S,
        X = X,
        A = A,
        ll = ll,
        inter_bonds = inter_bonds,
        intra_bonds = intra_bonds,
        out_path = out_path
    )

    def template_filter(cplx):
        cplx_desc.cplx = cplx
        return template.validate(cplx_desc)

    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(
        check_validity = check_validity,
        filters = [template_filter] if check_filters else None
    )

    if cplx is None or gen_mol is None:
        return None, None

    cplx_desc.cplx = cplx

    details = {
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': task.get_generated_seq()
    }

    return cplx_desc, details


def format_log(cplx_desc: ComplexDesc, details: dict, n: int):
    log = {
        'id': cplx_desc.id,
        'n': n,
        'pmetric': details['pmetric'],
        'smiles': details['smiles'],
        'gen_seq': details['gen_seq'],
        'tgt_chains': cplx_desc.tgt_chains,
        'lig_chains': cplx_desc.lig_chains,
        'gen_block_idx': cplx_desc.lig_block_ids,
    }
    for key, value in details.items():
        if key not in log:
            log[key] = value
    return log


def _get_first(config, names, default=None):
    for name in names:
        if name in config:
            return config[name]
    return default


def _option_enabled(config):
    return bool(config.get('enabled', config.get('merge', False)))


def _sdf_stem(path):
    path = Path(path)
    if path.name.endswith('.sdf'):
        return path.name[:-4]
    return path.stem


def _atom_name(symbol, counts):
    symbol = str(symbol).upper()
    counts[symbol] = counts.get(symbol, 0) + 1
    return f'{symbol}{counts[symbol]}'


def rdkit_atom_name(atom, counts):
    monomer_info = atom.GetMonomerInfo()
    if monomer_info is not None:
        name = monomer_info.GetName().strip()
        if name:
            return name
    if atom.HasProp('atom_name'):
        name = atom.GetProp('atom_name').strip()
        if name:
            return name
    return _atom_name(atom.GetSymbol(), counts)


def ligand_heavy_atom_refs(cplx, lig_chain):
    refs = []
    mol_idx = cplx.id2idx[lig_chain]
    for block_idx, block in enumerate(cplx[lig_chain]):
        for atom_idx, atom in enumerate(block):
            if str(atom.element).upper() in {'H', 'D'}:
                continue
            refs.append((mol_idx, block_idx, atom_idx))
    return refs


def remove_ligand_blocks_and_append_cap(cplx, lig_chain, remove_block_idxs, cap_block, cap_bonds, link_bonds):
    cplx = deepcopy(cplx)
    lig_mol_idx = cplx.id2idx[lig_chain]
    lig_mol = cplx[lig_chain]
    remove_block_idxs = set(int(idx) for idx in remove_block_idxs)
    cap_insert_idx = min(remove_block_idxs) if remove_block_idxs else len(lig_mol.blocks)

    kept_blocks = []
    block_old2new = {}
    for old_idx, block in enumerate(lig_mol.blocks):
        if old_idx == cap_insert_idx:
            cap_block_idx = len(kept_blocks)
            kept_blocks.append(cap_block)
        if old_idx in remove_block_idxs:
            continue
        block_old2new[old_idx] = len(kept_blocks)
        kept_blocks.append(block)

    if cap_insert_idx >= len(lig_mol.blocks):
        cap_block_idx = len(kept_blocks)
        kept_blocks.append(cap_block)

    molecules = []
    for mol_idx, mol in enumerate(cplx):
        if mol_idx == lig_mol_idx:
            molecules.append(Molecule(
                name=mol.name,
                blocks=kept_blocks,
                id=mol.id,
                properties=mol.properties,
            ))
        else:
            molecules.append(mol)

    bonds = []
    for bond in cplx.bonds:
        idx1 = tuple(bond.index1)
        idx2 = tuple(bond.index2)
        if idx1[0] == lig_mol_idx:
            if idx1[1] in remove_block_idxs:
                continue
            idx1 = (idx1[0], block_old2new[idx1[1]], idx1[2])
        if idx2[0] == lig_mol_idx:
            if idx2[1] in remove_block_idxs:
                continue
            idx2 = (idx2[0], block_old2new[idx2[1]], idx2[2])
        bonds.append(Bond(idx1, idx2, bond.bond_type))

    for atom_idx1, atom_idx2, bond_type in cap_bonds:
        bonds.append(Bond(
            index1=(lig_mol_idx, cap_block_idx, atom_idx1),
            index2=(lig_mol_idx, cap_block_idx, atom_idx2),
            bond_type=bond_type,
        ))

    for gen_ref, cap_atom_idx, bond_type in link_bonds:
        _mol_idx, block_idx, atom_idx = gen_ref
        bonds.append(Bond(
            index1=(lig_mol_idx, block_old2new[block_idx], atom_idx),
            index2=(lig_mol_idx, cap_block_idx, cap_atom_idx),
            bond_type=bond_type,
        ))

    return Complex(
        name=cplx.name,
        molecules=molecules,
        bonds=bonds,
        properties=cplx.properties,
    )


def merged_complex_with_cap_residue(cplx_desc, ref_mol, ref_indexes, gen_mol, gen_indexes, residue_name='CAP'):
    ref_mol, ref_old_to_new = heavy_atom_mol_with_map(ref_mol)
    gen_mol, gen_old_to_new = heavy_atom_mol_with_map(gen_mol)
    ref_indexes = remap_indexes(ref_indexes, ref_old_to_new, 'Reference')
    gen_indexes = remap_indexes(gen_indexes, gen_old_to_new, 'Generated')

    ref_to_gen_n = list(ref_indexes[:5])
    ref_to_gen_c = list(ref_indexes[5:])
    gen_n = list(gen_indexes[:5])
    gen_c = list(gen_indexes[5:])

    ref_work, gen_work = align_ref_to_generated(ref_mol, ref_indexes, gen_mol, gen_indexes)

    cplx = deepcopy(cplx_desc.cplx)
    lig_chain = cplx_desc.lig_chains[0]
    gen_refs = ligand_heavy_atom_refs(cplx, lig_chain)
    if len(gen_refs) != gen_work.GetNumAtoms():
        raise ValueError(
            f'Generated ligand atom count mismatch: complex has {len(gen_refs)} heavy atoms, '
            f'SDF has {gen_work.GetNumAtoms()}'
        )

    gen_conf = gen_work.GetConformer()
    for gen_idx, atom_ref in enumerate(gen_refs):
        pos = gen_conf.GetAtomPosition(gen_idx)
        cplx.get_atom(atom_ref).coordinate = [float(pos.x), float(pos.y), float(pos.z)]

    gen_remove = set(gen_n[:3] + gen_c[3:])
    ref_remove = set(ref_to_gen_n[3:] + ref_to_gen_c[:3])
    remove_block_idxs = {gen_refs[gen_idx][1] for gen_idx in gen_remove}

    ref_kept, ref_map = remove_atoms_with_map(ref_work, ref_remove)
    ref_conf = ref_kept.GetConformer()
    cap_name = str(residue_name)[:3]
    link_ref_atom_names = {
        ref_to_gen_n[1]: 'C',
        ref_to_gen_c[3]: 'N',
    }
    reserved_link_names = set(link_ref_atom_names.values())
    atom_counts = {}
    used_atom_names = set()
    cap_atoms = []
    for old_ref_idx, new_ref_idx in sorted(ref_map.items(), key=lambda item: item[1]):
        atom = ref_work.GetAtomWithIdx(old_ref_idx)
        pos = ref_conf.GetAtomPosition(new_ref_idx)
        if old_ref_idx in link_ref_atom_names:
            atom_name = link_ref_atom_names[old_ref_idx]
        else:
            atom_name = rdkit_atom_name(atom, atom_counts)
            if atom_name in reserved_link_names:
                atom_name = _atom_name(atom.GetSymbol(), atom_counts)
        while atom_name in used_atom_names:
            atom_name = _atom_name(atom.GetSymbol(), atom_counts)
        used_atom_names.add(atom_name)
        cap_atoms.append(Atom(
            name=atom_name,
            coordinate=[float(pos.x), float(pos.y), float(pos.z)],
            element=atom.GetSymbol(),
            id=str(new_ref_idx + 1),
        ))

    lig_mol = cplx[lig_chain]
    cap_res_id = lig_mol.blocks[min(remove_block_idxs)].id if remove_block_idxs else (1, '')
    cap_block = Block(
        name=cap_name,
        atoms=cap_atoms,
        id=cap_res_id,
        properties={'original_name': cap_name},
    )

    cap_bonds = []
    for bond in ref_kept.GetBonds():
        cap_bonds.append((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type_from_rdkit(bond),
        ))

    link_bonds = [
        (gen_refs[gen_n[3]], ref_map[ref_to_gen_n[1]], BondType.SINGLE),
        (gen_refs[gen_c[1]], ref_map[ref_to_gen_c[3]], BondType.SINGLE),
    ]
    return remove_ligand_blocks_and_append_cap(
        cplx,
        lig_chain,
        remove_block_idxs,
        cap_block,
        cap_bonds,
        link_bonds,
    )


def receptor_atoms_from_cplx(cplx, chain_ids):
    atoms = []
    for chain_id in chain_ids:
        if chain_id not in cplx.id2idx:
            continue
        for block in cplx[chain_id]:
            for atom in block:
                symbol = str(atom.element).upper()
                if symbol in {'H', 'D'}:
                    continue
                atoms.append((symbol, atom.coordinate))
    return atoms


def _zero_based_indexes(indexes, index_base):
    return [int(index) - int(index_base) for index in indexes]


def _coord_distance(coord1, coord2):
    return sum((float(coord1[i]) - float(coord2[i])) ** 2 for i in range(3)) ** 0.5


def _atom_by_name(block, atom_name):
    atom_name = str(atom_name).strip()
    for atom in block:
        if str(atom.name).strip() == atom_name:
            return atom
    return None


def _format_ligand_amide_violation(violation):
    return (
        f'{violation["chain"]}:{violation["prev_block"]}->{violation["next_block"]} '
        f'C-N={violation["distance"]:.3f}A'
    )


def find_ligand_amide_bond_violations(cplx, lig_chains, max_distance):
    violations = []
    max_distance = float(max_distance)
    for lig_chain in lig_chains:
        if lig_chain not in cplx.id2idx:
            continue
        blocks = list(cplx[lig_chain])
        for prev_block, next_block in zip(blocks[:-1], blocks[1:]):
            c_atom = _atom_by_name(prev_block, 'C')
            n_atom = _atom_by_name(next_block, 'N')
            if c_atom is None or n_atom is None:
                continue
            distance = _coord_distance(c_atom.coordinate, n_atom.coordinate)
            if distance > max_distance:
                violations.append({
                    'chain': lig_chain,
                    'prev_block': f'{prev_block.name}{prev_block.id}',
                    'next_block': f'{next_block.name}{next_block.id}',
                    'distance': distance,
                    'max_distance': max_distance,
                })
    return violations


def _merge_amide_check_options(config, template):
    opt = dict(config.get('amide_bond_check', {}) or {})
    opt.update(dict(template.pop('amide_bond_check', {}) or {}))
    opt.setdefault('enabled', True)
    opt.setdefault('max_c_n_distance', opt.get('max_distance', 1.8))
    return opt


def _amide_check_enabled(amide_check_opt):
    return bool(amide_check_opt.get('enabled', True))


def reject_implausible_ligand_amides(cplx_desc, details, n, amide_check_opt):
    if details is None or not _amide_check_enabled(amide_check_opt):
        return details
    max_distance = float(amide_check_opt.get('max_c_n_distance', amide_check_opt.get('max_distance', 1.8)))
    violations = find_ligand_amide_bond_violations(cplx_desc.cplx, cplx_desc.lig_chains, max_distance)
    if not violations:
        return details
    joined = '; '.join(_format_ligand_amide_violation(item) for item in violations[:3])
    if len(violations) > 3:
        joined += f'; +{len(violations) - 3} more'
    return _pipeline_reject(f'implausible ligand amide bond(s): {joined}', cplx_desc, n)


def _pipeline_reject(reason, cplx_desc, n):
    print_log(f'Rejected {cplx_desc.id}/{n}: {reason}', level='DEBUG')
    return None


def compute_terminal_rmsd_details(cplx_desc, raw_sdf_path, pipeline_opt):
    control = cplx_desc.control or {}
    ref_path = control.get('term_rel_pos_sdf_path', None)
    ref_atom_indices = control.get('term_rel_pos_atom_indices', None)
    if ref_path is None or ref_atom_indices is None:
        return {}

    ref_indexes = _zero_based_indexes(
        ref_atom_indices,
        control.get('term_rel_pos_index_base', 0),
    )
    ref_mol = load_sdf_mol(ref_path, control.get('term_rel_pos_mol_index', 0))
    gen_mol = load_sdf_mol(
        raw_sdf_path,
        int(_get_first(pipeline_opt, ('mol_index', 'gen_mol_index'), 0)),
    )
    gen_atom_indices = control.get('gen_atom_indices', pipeline_opt.get('gen_atom_indices', None))
    if gen_atom_indices is None:
        gen_indexes = infer_generated_indexes(gen_mol)
    else:
        gen_indexes = _zero_based_indexes(
            gen_atom_indices,
            control.get('gen_index_base', pipeline_opt.get('gen_index_base', 0)),
        )

    ref_coords = get_positions(ref_mol, ref_indexes)
    gen_coords = get_positions(gen_mol, gen_indexes)
    aligned = kabsch_align(gen_coords, ref_coords)
    terminal_rmsd = rmsd(aligned, ref_coords)
    return {
        'rmsd': terminal_rmsd,
        'terminal_rmsd': terminal_rmsd,
        'n_term_rmsd': rmsd(aligned[:5], ref_coords[:5]),
        'c_term_rmsd': rmsd(aligned[5:], ref_coords[5:]),
        'reference_sdf': str(ref_path),
        'conformer': _sdf_stem(ref_path),
    }


def add_terminal_rmsd_details(cplx_desc, details, raw_sdf_path, pipeline_opt):
    try:
        details.update(compute_terminal_rmsd_details(cplx_desc, raw_sdf_path, pipeline_opt))
    except Exception as exc:
        details['rmsd_error'] = str(exc)
    return details


PIPELINE_METADATA_FIELDS = [
    'accepted_name',
    'accepted_cif',
    'id',
    'sample_slot',
    'conformer',
    'rmsd',
    'reference_sdf',
    'terminal_rmsd',
    'n_term_rmsd',
    'c_term_rmsd',
    'rmsd_cutoff',
    'self_clash_count',
    'receptor_clash_count',
    'raw_pdb',
    'raw_sdf',
    'gen_seq',
    'smiles',
]


def append_pipeline_metadata(accepted_dir, row):
    path = Path(accepted_dir) / 'metadata.csv'
    with path.open('a+', newline='') as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        write_header = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_METADATA_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in PIPELINE_METADATA_FIELDS})
        handle.flush()
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def resolve_pipeline_accepted_dir(save_dir, pipeline_opt):
    output_dir = pipeline_opt.get('output_dir', None)
    if output_dir is None:
        return os.path.join(os.path.dirname(save_dir), 'accepted')
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(os.path.dirname(save_dir), output_dir)


def _pipeline_output_dir_name(pipeline_opt):
    output_dir = pipeline_opt.get('output_dir', None)
    if output_dir is None or os.path.isabs(output_dir):
        return 'accepted'
    return output_dir


def _read_pipeline_metadata(accepted_dir):
    metadata_path = Path(accepted_dir) / 'metadata.csv'
    if not metadata_path.exists():
        return []
    with metadata_path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _accepted_index(name):
    match = re.search(r'accepted_(\d+)$', str(name or ''))
    return None if match is None else int(match.group(1))


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _existing_pipeline_state(accepted_dir, target_ids):
    accepted_dir = Path(accepted_dir)
    rows = _read_pipeline_metadata(accepted_dir)
    counts_by_id = {target_id: 0 for target_id in target_ids}
    sample_slot_max_by_id = {target_id: -1 for target_id in target_ids}
    max_accepted_index = 0

    for row in rows:
        accepted_name = row.get('accepted_name', '')
        if not accepted_name and row.get('accepted_cif', ''):
            accepted_name = Path(row['accepted_cif']).stem
        index = _accepted_index(accepted_name)
        if index is not None:
            max_accepted_index = max(max_accepted_index, index)

        target_id = row.get('id', '')
        if target_id not in counts_by_id and len(target_ids) == 1:
            target_id = target_ids[0]
        if target_id in counts_by_id:
            counts_by_id[target_id] += 1
            sample_slot = _safe_int(row.get('sample_slot', None))
            if sample_slot is not None:
                sample_slot_max_by_id[target_id] = max(
                    sample_slot_max_by_id[target_id],
                    sample_slot,
                )

    for cif_path in accepted_dir.glob('accepted_*.cif'):
        index = _accepted_index(cif_path.stem)
        if index is not None:
            max_accepted_index = max(max_accepted_index, index)
    if not rows and len(target_ids) == 1:
        counts_by_id[target_ids[0]] = len(list(accepted_dir.glob('accepted_*.cif')))

    return {
        'rows': rows,
        'counts_by_id': counts_by_id,
        'sample_slot_max_by_id': sample_slot_max_by_id,
        'max_accepted_index': max_accepted_index,
    }


def _max_candidate_slot(candidate_dir, target_id):
    target_dir = Path(candidate_dir) / target_id
    max_slot = -1
    if not target_dir.exists():
        return max_slot
    for pdb_path in target_dir.glob('*.pdb'):
        slot = _safe_int(pdb_path.stem)
        if slot is not None:
            max_slot = max(max_slot, slot)
    return max_slot


def _resume_options(config, pipeline_opt):
    raw_global = config.get('resume_generation', None)
    raw_pipeline = pipeline_opt.get('resume_generation', None)
    options = {}
    for raw in (raw_global, raw_pipeline):
        if raw is None:
            continue
        if isinstance(raw, dict):
            options.update(raw)
        else:
            options['enabled'] = bool(raw)
    for source in (config, pipeline_opt):
        for key in ('previous_dir', 'resume_dir', 'resume_from'):
            if key in source:
                options[key] = source[key]
    options['enabled'] = bool(options.get('enabled', False))
    return options


def _sanitize_options(config, pipeline_opt, amide_check_opt):
    options = {}
    for key in ('sanitize_generation', 'santinize_generation', 'sanitize', 'santinize'):
        for source in (config, pipeline_opt):
            if key not in source:
                continue
            raw = source[key]
            if isinstance(raw, dict):
                options.update(raw)
            else:
                options['enabled'] = bool(raw)
    options['enabled'] = bool(options.get('enabled', False))
    options.setdefault(
        'max_c_n_distance',
        amide_check_opt.get('max_c_n_distance', amide_check_opt.get('max_distance', 1.8)),
    )
    return options


def _resume_previous_dir(resume_opt, save_root):
    previous_dir = _get_first(
        resume_opt,
        ('previous_dir', 'resume_dir', 'resume_from', 'path', 'dir'),
        None,
    )
    if previous_dir is None:
        previous_dir = save_root
    previous_root = Path(previous_dir)
    if not previous_root.exists():
        raise ValueError(f'resume_generation previous_dir does not exist: {previous_dir}')
    return previous_root


def _resume_accepted_dir(previous_dir, run_name, pipeline_opt):
    root = Path(previous_dir)
    output_dir = _pipeline_output_dir_name(pipeline_opt)
    candidates = []
    configured_output_dir = pipeline_opt.get('output_dir', None)
    if configured_output_dir is not None and os.path.isabs(configured_output_dir):
        candidates.append(Path(configured_output_dir))
    candidates.extend([
        root / run_name / output_dir,
        root / output_dir,
        root,
    ])
    for candidate in candidates:
        if (candidate / 'metadata.csv').exists() or any(candidate.glob('accepted_*.cif')):
            return candidate
    return candidates[0]


def _dataset_lig_chain_ids(dataset):
    lig_chain_ids = []
    for chain_spec in dataset.lig_chains:
        for chain_id in chain_spec:
            if chain_id not in lig_chain_ids:
                lig_chain_ids.append(chain_id)
    return lig_chain_ids


def _dataset_item_target_ids(dataset):
    sample_repeat = max(1, int(getattr(dataset, 'n_samples', 1)))
    ids = []
    for item_idx in range(len(dataset)):
        pdb_idx = min(item_idx // sample_repeat, len(dataset.pdb_paths) - 1)
        ids.append(Path(dataset.pdb_paths[pdb_idx]).stem)
    return ids


def build_pipeline_resume_state(
        resume_opt,
        save_root,
        run_name,
        candidate_dir,
        pipeline_opt,
        dataset,
        n_samples,
    ):
    if not resume_opt.get('enabled', False):
        return None

    previous_root = _resume_previous_dir(resume_opt, save_root)

    target_ids = _dataset_item_target_ids(dataset)
    accepted_dir = _resume_accepted_dir(previous_root, run_name, pipeline_opt)
    existing = _existing_pipeline_state(accepted_dir, target_ids)
    counts_by_id = existing['counts_by_id']
    sample_slot_max_by_id = existing['sample_slot_max_by_id']

    goal_by_item, successes_by_item, waiting_list, next_sample_index_by_item = [], [], [], []
    total_remaining = 0
    for item_idx, target_id in enumerate(target_ids):
        existing_count = counts_by_id.get(target_id, 0)
        clipped_existing = min(existing_count, n_samples)
        remaining = max(0, n_samples - clipped_existing)
        goal_by_item.append(n_samples)
        successes_by_item.append(clipped_existing)
        next_slot = max(
            n_samples,
            sample_slot_max_by_id.get(target_id, -1) + 1,
            _max_candidate_slot(candidate_dir, target_id) + 1,
        )
        for offset in range(remaining):
            waiting_list.append((item_idx, next_slot + offset))
        next_sample_index_by_item.append(next_slot + remaining)
        total_remaining += remaining
        print_log(
            f'Resume target {run_name}/{target_id}: '
            f'existing={existing_count}, goal={n_samples}, to_generate={remaining}',
            level='INFO',
        )

    print_log(
        f'Resume generation for {run_name}: previous accepted dir={accepted_dir}, '
        f'total_to_generate={total_remaining}',
        level='INFO',
    )
    return {
        'accepted_start_index': existing['max_accepted_index'] + 1,
        'append_results': True,
        'goal_by_item': goal_by_item,
        'successes_by_item': successes_by_item,
        'waiting_list': waiting_list,
        'next_sample_index_by_item': next_sample_index_by_item,
        'start_successes': sum(successes_by_item),
        'remaining': total_remaining,
        'previous_accepted_dir': str(accepted_dir),
    }


def _resolve_metadata_cif_path(row, accepted_dir):
    accepted_dir = Path(accepted_dir)
    candidates = []
    accepted_cif = row.get('accepted_cif', '')
    if accepted_cif:
        accepted_cif_path = Path(accepted_cif)
        candidates.append(accepted_dir / accepted_cif_path.name)
        if not accepted_cif_path.is_absolute():
            candidates.append(accepted_dir / accepted_cif_path)
        else:
            candidates.append(accepted_cif_path)
    accepted_name = row.get('accepted_name', '')
    if accepted_name:
        candidates.append(accepted_dir / f'{accepted_name}.cif')
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _accepted_metadata_entries(accepted_dir):
    accepted_dir = Path(accepted_dir)
    entries = []
    seen_paths = set()
    for row in _read_pipeline_metadata(accepted_dir):
        cif_path = _resolve_metadata_cif_path(row, accepted_dir)
        if cif_path is None:
            entries.append({'row': row, 'cif_path': None})
            continue
        cif_path = cif_path.resolve()
        seen_paths.add(cif_path)
        entries.append({'row': row, 'cif_path': cif_path})

    for cif_path in accepted_dir.glob('accepted_*.cif'):
        resolved = cif_path.resolve()
        if resolved in seen_paths:
            continue
        entries.append({
            'row': {
                'accepted_name': cif_path.stem,
                'accepted_cif': str(cif_path),
            },
            'cif_path': resolved,
        })

    def sort_key(entry):
        row = entry['row']
        name = row.get('accepted_name', '')
        if not name and entry['cif_path'] is not None:
            name = entry['cif_path'].stem
        index = _accepted_index(name)
        return (index is None, index if index is not None else name)

    return sorted(entries, key=sort_key)


def _write_pipeline_metadata(accepted_dir, rows):
    path = Path(accepted_dir) / 'metadata.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in PIPELINE_METADATA_FIELDS})


def _sanitize_lock_path(accepted_dir):
    return Path(accepted_dir) / '.sanitize_generation.lock'


def sanitize_pipeline_accepted_dir(accepted_dir, lig_chains, sanitize_opt):
    accepted_dir = Path(accepted_dir)
    if not accepted_dir.exists():
        print_log(f'Sanitize generation: accepted dir does not exist, skipping {accepted_dir}', level='INFO')
        return {'kept': 0, 'removed': 0}

    lock_path = _sanitize_lock_path(accepted_dir)
    lock_path.touch(exist_ok=True)
    lock_handle = lock_path.open('r+')
    try:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return _sanitize_pipeline_accepted_dir_unlocked(accepted_dir, lig_chains, sanitize_opt)
    finally:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _sanitize_pipeline_accepted_dir_unlocked(accepted_dir, lig_chains, sanitize_opt):
    max_distance = float(sanitize_opt.get('max_c_n_distance', sanitize_opt.get('max_distance', 1.8)))
    entries = _accepted_metadata_entries(accepted_dir)
    kept = []
    removed = []

    for entry in entries:
        row = dict(entry['row'])
        cif_path = entry['cif_path']
        if cif_path is None or not Path(cif_path).exists():
            removed.append((cif_path, 'missing cif file'))
            continue
        try:
            cplx = cif_to_complex(str(cif_path), selected_chains=lig_chains)
            violations = find_ligand_amide_bond_violations(cplx, lig_chains, max_distance)
        except Exception as exc:
            print_log(
                f'Sanitize generation: keeping {Path(cif_path).name}; '
                f'could not parse/check CIF: {exc}',
                level='WARN',
            )
            kept.append({'row': row, 'cif_path': Path(cif_path)})
            continue
        if violations:
            reason = '; '.join(_format_ligand_amide_violation(item) for item in violations[:3])
            removed.append((Path(cif_path), reason))
            continue
        kept.append({'row': row, 'cif_path': Path(cif_path)})

    for cif_path, reason in removed:
        if cif_path is None:
            print_log(f'Sanitize generation: dropped metadata row with missing CIF ({reason})', level='INFO')
            continue
        print_log(f'Sanitize generation: removing {cif_path.name}: {reason}', level='INFO')
        if cif_path.exists():
            cif_path.unlink()
        sdf_path = cif_path.with_suffix('.sdf')
        if sdf_path.exists():
            sdf_path.unlink()

    temp_entries = []
    for entry in kept:
        old_cif = entry['cif_path']
        temp_cif = accepted_dir / f'.sanitize_tmp_{old_cif.name}'
        if temp_cif.exists():
            temp_cif.unlink()
        shutil.move(str(old_cif), str(temp_cif))
        old_sdf = old_cif.with_suffix('.sdf')
        temp_sdf = None
        if old_sdf.exists():
            temp_sdf = accepted_dir / f'.sanitize_tmp_{old_sdf.name}'
            if temp_sdf.exists():
                temp_sdf.unlink()
            shutil.move(str(old_sdf), str(temp_sdf))
        temp_entries.append({
            'row': entry['row'],
            'temp_cif': temp_cif,
            'temp_sdf': temp_sdf,
        })

    metadata_rows = []
    for new_index, entry in enumerate(temp_entries, start=1):
        accepted_name = f'accepted_{new_index}'
        new_cif = accepted_dir / f'{accepted_name}.cif'
        shutil.move(str(entry['temp_cif']), str(new_cif))
        if entry['temp_sdf'] is not None:
            shutil.move(str(entry['temp_sdf']), str(accepted_dir / f'{accepted_name}.sdf'))
        row = dict(entry['row'])
        row['accepted_name'] = accepted_name
        row['accepted_cif'] = str(new_cif)
        metadata_rows.append(row)

    _write_pipeline_metadata(accepted_dir, metadata_rows)
    print_log(
        f'Sanitize generation: kept={len(metadata_rows)}, removed={len(removed)}, '
        f'accepted_dir={accepted_dir}',
        level='INFO',
    )
    return {'kept': len(metadata_rows), 'removed': len(removed)}


def _resolve_split_options(config, args):
    split_opt = dict(config.get('task_split', config.get('generation_split', {})) or {})
    total = _get_first(split_opt, ('total', 'total_gpus', 'num_shards', 'world_size'), 1)
    index = _get_first(split_opt, ('index', 'gpu_id', 'shard_id', 'rank'), 0)
    if args.num_shards is not None:
        total = args.num_shards
    if args.shard_id is not None:
        index = args.shard_id

    total = int(total)
    index = int(index)
    enabled = bool(split_opt.get('enabled', False)) or total > 1
    if not enabled:
        total, index = 1, 0
    if total <= 0:
        raise ValueError('task_split.total/--num_shards must be a positive integer')
    if index < 0 or index >= total:
        raise ValueError(f'task_split.index/--shard_id must be in [0, {total - 1}], got {index}')
    return {
        'enabled': total > 1,
        'total': total,
        'index': index,
    }


def _default_generation_state(dataset, n_samples):
    return {
        'accepted_start_index': 1,
        'append_results': False,
        'goal_by_item': [n_samples for _ in range(len(dataset))],
        'successes_by_item': [0 for _ in range(len(dataset))],
        'waiting_list': [
            (item_idx, n)
            for n in range(n_samples)
            for item_idx in range(len(dataset))
        ],
        'next_sample_index_by_item': [n_samples for _ in range(len(dataset))],
        'start_successes': 0,
        'remaining': n_samples * len(dataset),
    }


def apply_generation_shard(resume_state, split_opt, dataset, n_samples, run_name):
    if not split_opt.get('enabled', False):
        return resume_state

    total = split_opt['total']
    index = split_opt['index']
    state = _default_generation_state(dataset, n_samples) if resume_state is None else dict(resume_state)
    original_waiting_list = list(state.get('waiting_list', []))
    sharded_waiting_list = [
        slot
        for pos, slot in enumerate(original_waiting_list)
        if pos % total == index
    ]
    assigned_by_item = [0 for _ in range(len(dataset))]
    for item_idx, _ in sharded_waiting_list:
        assigned_by_item[item_idx] += 1

    base_next_sample_index = state.get(
        'next_sample_index_by_item',
        [n_samples for _ in range(len(dataset))],
    )
    state.update({
        'accepted_start_index': int(state.get('accepted_start_index', 1)) + index,
        'accepted_index_stride': total,
        'append_results': state.get('append_results', False),
        'goal_by_item': assigned_by_item,
        'successes_by_item': [0 for _ in range(len(dataset))],
        'waiting_list': sharded_waiting_list,
        'next_sample_index_by_item': [
            int(value) + index
            for value in base_next_sample_index
        ],
        'sample_index_stride': total,
        'results_name': f'results.shard{index}of{total}.jsonl',
        'tmp_dir_name': f'tmp_shard{index}of{total}',
        'start_successes': 0,
        'remaining': sum(assigned_by_item),
    })
    print_log(
        f'Generation shard for {run_name}: shard={index}/{total}, '
        f'assigned_to_generate={state["remaining"]}, '
        f'total_to_generate={len(original_waiting_list)}',
        level='INFO',
    )
    return state


def _shard_max_sampling(max_sampling, split_opt):
    if max_sampling is None or not split_opt.get('enabled', False):
        return max_sampling
    total = split_opt['total']
    index = split_opt['index']
    return (int(max_sampling) + total - 1 - index) // total


def _resolve_prefetch_options(config, args):
    prefetch_opt = dict(config.get('generation_prefetch', {}) or {})
    if 'num_workers' not in prefetch_opt:
        prefetch_opt['num_workers'] = config.get('num_workers', 0)
    if args.num_workers is not None:
        prefetch_opt['num_workers'] = args.num_workers
    prefetch_opt.setdefault('prefetch_batches', config.get('prefetch_batches', 2))
    prefetch_opt.setdefault('backend', 'thread')
    prefetch_opt.setdefault('start_method', 'spawn')
    prefetch_opt.setdefault('torch_num_threads', 1)
    prefetch_opt['num_workers'] = int(prefetch_opt.get('num_workers', 0))
    prefetch_opt['prefetch_batches'] = int(prefetch_opt.get('prefetch_batches', 2))
    prefetch_opt['backend'] = str(prefetch_opt.get('backend', 'thread')).lower()
    if prefetch_opt['backend'] == 'process' and not bool(prefetch_opt.get('allow_process_backend', False)):
        print_log(
            'generation_prefetch.backend=process was requested, but process prefetch '
            'can terminate GPU jobs on large datasets because every spawned worker '
            'needs its own Python/dataset state. Using backend=thread instead. '
            'Set generation_prefetch.allow_process_backend=true to force process mode.',
            level='WARN',
        )
        prefetch_opt['backend'] = 'thread'
    if prefetch_opt['num_workers'] < 0:
        raise ValueError('generation_prefetch.num_workers must be >= 0')
    if prefetch_opt['prefetch_batches'] <= 0:
        raise ValueError('generation_prefetch.prefetch_batches must be positive')
    if prefetch_opt['backend'] not in {'process', 'thread'}:
        raise ValueError('generation_prefetch.backend must be "process" or "thread"')
    return prefetch_opt


def _uses_without_replacement_sampler(dataset):
    sampler = getattr(getattr(dataset, 'config', None), 'term_rel_pos_sampler', None)
    return sampler is not None and not getattr(sampler, 'sample_with_replacement', True)


def _generation_stats_path(save_dir, results_name):
    stem = Path(results_name).stem
    suffix = '' if stem == 'results' else stem[len('results'):] if stem.startswith('results') else f'.{stem}'
    return Path(save_dir) / f'generation_stats{suffix}.json'


def _write_generation_stats(stats_path, recorder, batches_generated, accepted_start):
    generated = int(recorder.num_generated)
    accepted = int(recorder.num_successful - accepted_start)
    stats = {
        'batches_generated': int(batches_generated),
        'generated_candidates': generated,
        'failed_candidates': int(recorder.num_failed),
        'accepted': accepted,
        'accepted_total': int(recorder.num_successful),
        'target_total': int(recorder.total),
        'accept_rate': accepted / generated if generated else 0.0,
    }
    stats_path = Path(stats_path)
    tmp_path = stats_path.with_suffix(stats_path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(stats, indent=2) + '\n')
    os.replace(tmp_path, stats_path)
    return stats


def _report_generation_progress(stats_path, recorder, batches_generated, accepted_start):
    stats = _write_generation_stats(stats_path, recorder, batches_generated, accepted_start)
    message = (
        f'{stats["batches_generated"]} batches generated '
        f'({stats["generated_candidates"]} designs in total), '
        f'{stats["accepted"]} accepted'
    )
    if accepted_start:
        message += f' this run ({stats["accepted_total"]}/{stats["target_total"]} total accepted)'
    message += f', accept_rate={stats["accept_rate"]:.4f}'
    print_log(message, level='INFO')


def _resolve_postprocess_options(config, args):
    postprocess_opt = dict(config.get('generation_postprocess', {}) or {})
    if 'num_workers' not in postprocess_opt:
        postprocess_opt['num_workers'] = config.get('postprocess_workers', 0)
    if args.postprocess_workers is not None:
        postprocess_opt['num_workers'] = args.postprocess_workers
    postprocess_opt.setdefault('backend', 'process')
    postprocess_opt.setdefault('start_method', 'spawn')
    postprocess_opt.setdefault('torch_num_threads', 1)
    postprocess_opt['num_workers'] = int(postprocess_opt.get('num_workers', 0))
    postprocess_opt['backend'] = str(postprocess_opt.get('backend', 'process')).lower()
    if postprocess_opt['num_workers'] < 0:
        raise ValueError('generation_postprocess.num_workers must be >= 0')
    if postprocess_opt['backend'] not in {'process', 'thread'}:
        raise ValueError('generation_postprocess.backend must be "process" or "thread"')
    return postprocess_opt


def _make_postprocess_executor(postprocess_opt):
    num_workers = int(postprocess_opt.get('num_workers', 0))
    if num_workers <= 0:
        return None
    backend = str(postprocess_opt.get('backend', 'process')).lower()
    torch_num_threads = postprocess_opt.get('torch_num_threads', 1)
    if backend == 'thread':
        return ThreadPoolExecutor(
            max_workers=num_workers,
            initializer=_init_postprocess_worker,
            initargs=(torch_num_threads,),
        )
    mp_context = mp.get_context(postprocess_opt.get('start_method', 'spawn'))
    return ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_postprocess_worker,
        initargs=(torch_num_threads,),
        mp_context=mp_context,
    )


def _pipeline_tmp_dir(accepted_dir, resume_state):
    tmp_dir_name = resume_state.get('tmp_dir_name', 'tmp')
    return Path(accepted_dir) / f'.postprocess_{tmp_dir_name}'


def _cleanup_pipeline_temp(log):
    if not log:
        return
    for key in ('accepted_cif', 'merged_sdf'):
        path = log.get(key, '')
        if path:
            path = Path(path)
            if path.exists():
                path.unlink()


def _finalize_pipeline_temp(log, accepted_dir, accepted_index):
    accepted_dir = Path(accepted_dir)
    temp_cif = Path(log.get('accepted_cif', ''))
    if not temp_cif.exists():
        raise FileNotFoundError(f'Accepted pipeline temp CIF does not exist: {temp_cif}')

    accepted_name = f'accepted_{accepted_index}'
    final_cif = accepted_dir / f'{accepted_name}.cif'
    shutil.move(str(temp_cif), str(final_cif))
    log['accepted_name'] = accepted_name
    log['accepted_cif'] = str(final_cif)

    temp_sdf_value = log.get('merged_sdf', '')
    if temp_sdf_value:
        temp_sdf = Path(temp_sdf_value)
        if temp_sdf.exists():
            final_sdf = accepted_dir / f'{accepted_name}.sdf'
            shutil.move(str(temp_sdf), str(final_sdf))
            log['merged_sdf'] = str(final_sdf)

    append_pipeline_metadata(accepted_dir, {
        **log,
        'id': log.get('id', ''),
        'sample_slot': log.get('n', ''),
    })
    return log


def _postprocess_candidate_worker(task):
    item_idx = task['item_idx']
    n = task['n']
    cplx_desc = task['cplx_desc']
    try:
        tmp_target_dir = task['tmp_target_dir']
        target_dir = task['target_dir']
        pocket_path = task['pocket_path']
        os.makedirs(tmp_target_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        if task.get('write_pocket', False) and not os.path.exists(pocket_path):
            complex_to_pdb(cplx_desc.cplx, pocket_path, cplx_desc.tgt_chains)

        cplx_desc, details = data_to_cplx(
            cplx_desc,
            task['template_config'],
            task['S'],
            task['X'],
            task['A'],
            task['ll'],
            task['inter_bonds'],
            task['intra_bonds'],
            task['out_path'],
            check_validity=task.get('check_validity', False),
            check_filters=task.get('check_filters', False),
        )
        if cplx_desc is None:
            return {'item_idx': item_idx, 'n': n, 'log': None}

        if task.get('final_cycle', True):
            details = reject_implausible_ligand_amides(
                cplx_desc,
                details,
                n,
                task.get('amide_check_opt', {}),
            )

        if details is not None and task.get('pipeline_enabled', False):
            try:
                temp_name = f'tmp_{os.getpid()}_{item_idx}_{n}'
                details = run_terminal_merge_pipeline(
                    cplx_desc,
                    details,
                    n,
                    task['out_path'],
                    task['out_path'].rstrip('.pdb') + '.sdf',
                    task['accepted_tmp_dir'],
                    task.get('pipeline_opt', {}),
                    accepted_name=temp_name,
                    append_metadata_row=False,
                )
            except Exception as exc:
                details = _pipeline_reject(f'pipeline error: {exc}', cplx_desc, n)
        elif details is not None and task.get('final_cycle', True):
            details = add_terminal_rmsd_details(
                cplx_desc,
                details,
                task['out_path'].rstrip('.pdb') + '.sdf',
                task.get('pipeline_opt', {}),
            )

        log = None if details is None else format_log(cplx_desc, details, n)
        return {'item_idx': item_idx, 'n': n, 'log': log}
    except Exception as exc:
        print_log(f'Post-processing failed for {cplx_desc.id}/{n}: {exc}', level='WARN')
        return {'item_idx': item_idx, 'n': n, 'log': None}


def _run_postprocess_tasks(tasks, executor):
    if executor is None:
        return [_postprocess_candidate_worker(task) for task in tasks]

    futures = [executor.submit(_postprocess_candidate_worker, task) for task in tasks]
    results = []
    for task, future in zip(tasks, futures):
        try:
            results.append(future.result())
        except Exception as exc:
            print_log(
                f'Post-processing worker crashed for item={task["item_idx"]}, '
                f'n={task["n"]}; falling back to serial post-processing: {exc}',
                level='WARN',
            )
            results.append(_postprocess_candidate_worker(task))
    return results


def run_terminal_merge_pipeline(
        cplx_desc,
        details,
        n,
        raw_pdb_path,
        raw_sdf_path,
        accepted_dir,
        pipeline_opt,
        accepted_index=None,
        accepted_name=None,
        append_metadata_row=True,
    ):
    control = cplx_desc.control or {}
    rmsd_details = compute_terminal_rmsd_details(cplx_desc, raw_sdf_path, pipeline_opt)
    ref_path = rmsd_details.get('reference_sdf', None)
    if ref_path is None:
        raise ValueError('pipeline.enabled requires term_rel_pos_sdf so each sample has a reference SDF')

    ref_atom_indices = control.get('term_rel_pos_atom_indices', None)
    ref_index_base = control.get('term_rel_pos_index_base', 0)
    ref_mol_index = control.get('term_rel_pos_mol_index', 0)
    gen_mol_index = int(_get_first(pipeline_opt, ('mol_index', 'gen_mol_index'), 0))
    ref_indexes = _zero_based_indexes(ref_atom_indices, ref_index_base)

    ref_mol = load_sdf_mol(ref_path, ref_mol_index)
    gen_mol = load_sdf_mol(raw_sdf_path, gen_mol_index)
    gen_atom_indices = control.get('gen_atom_indices', pipeline_opt.get('gen_atom_indices', None))
    if gen_atom_indices is None:
        gen_indexes = infer_generated_indexes(gen_mol)
    else:
        gen_indexes = _zero_based_indexes(
            gen_atom_indices,
            control.get('gen_index_base', pipeline_opt.get('gen_index_base', 0)),
        )

    ref_coords = get_positions(ref_mol, ref_indexes)
    gen_coords = get_positions(gen_mol, gen_indexes)
    aligned = kabsch_align(gen_coords, ref_coords)
    terminal_rmsd = rmsd(aligned, ref_coords)
    n_term_rmsd = rmsd(aligned[:5], ref_coords[:5])
    c_term_rmsd = rmsd(aligned[5:], ref_coords[5:])
    rmsd_cutoff = float(_get_first(pipeline_opt, ('rmsd_cutoff', 'merge_rmsd_cutoff'), 0.5))
    if terminal_rmsd >= rmsd_cutoff:
        return _pipeline_reject(f'terminal RMSD {terminal_rmsd:.4f} >= cutoff {rmsd_cutoff:.4f}', cplx_desc, n)

    merged_mol = merge_terminal_cap(ref_mol, ref_indexes, gen_mol, gen_indexes)
    clash_threshold = float(pipeline_opt.get('clash_threshold', 0.8))
    self_clashes = []
    receptor_clashes = []
    if not bool(pipeline_opt.get('no_clash_check', False)):
        self_clashes = detect_clashes(merged_mol, threshold=clash_threshold)
        receptor_atoms = receptor_atoms_from_cplx(cplx_desc.cplx, cplx_desc.tgt_chains)
        receptor_clashes = detect_receptor_clashes(merged_mol, receptor_atoms, threshold=clash_threshold)
        if self_clashes or receptor_clashes:
            return _pipeline_reject(
                f'clash self={len(self_clashes)} receptor={len(receptor_clashes)}',
                cplx_desc,
                n,
            )

    residue_name = pipeline_opt.get('ligand_residue_name', 'CAP')
    accepted_dir = Path(accepted_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    if accepted_name is None:
        accepted_name = f'accepted_{accepted_index}'
    cif_path = accepted_dir / f'{accepted_name}.cif'
    merged_cplx = merged_complex_with_cap_residue(
        cplx_desc,
        ref_mol,
        ref_indexes,
        gen_mol,
        gen_indexes,
        residue_name=residue_name,
    )
    complex_to_cif(
        merged_cplx,
        str(cif_path),
        selected_chains=cplx_desc.tgt_chains + cplx_desc.lig_chains,
        title=accepted_name,
    )

    merged_sdf_path = ''
    if bool(pipeline_opt.get('write_merged_sdf', False)):
        merged_sdf_path = str(accepted_dir / f'{accepted_name}.sdf')
        write_sdf(merged_mol, merged_sdf_path)

    details.update({
        'accepted_name': accepted_name,
        **rmsd_details,
        'rmsd_cutoff': rmsd_cutoff,
        'self_clash_count': len(self_clashes),
        'receptor_clash_count': len(receptor_clashes),
        'raw_pdb': str(raw_pdb_path),
        'raw_sdf': str(raw_sdf_path),
        'accepted_cif': str(cif_path),
        'merged_sdf': merged_sdf_path,
        'merge_status': 'saved',
    })
    if append_metadata_row:
        append_pipeline_metadata(accepted_dir, {
            **details,
            'id': cplx_desc.id,
            'sample_slot': n,
        })
    return details


def generate_for_one_template(
        model,
        dataset,
        n_samples,
        batch_size,
        save_dir,
        device,
        sample_opt,
        pipeline_opt=None,
        max_sampling=None,
        resume_state=None,
        amide_check_opt=None,
        prefetch_opt=None,
        prefetch_seed=None,
        postprocess_opt=None,
    ):
    pipeline_opt = {} if pipeline_opt is None else dict(pipeline_opt)
    resume_state = {} if resume_state is None else dict(resume_state)
    amide_check_opt = {} if amide_check_opt is None else dict(amide_check_opt)
    prefetch_opt = {} if prefetch_opt is None else dict(prefetch_opt)
    postprocess_opt = {} if postprocess_opt is None else dict(postprocess_opt)
    accepted_dir = resolve_pipeline_accepted_dir(save_dir, pipeline_opt)
    if _option_enabled(pipeline_opt):
        os.makedirs(accepted_dir, exist_ok=True)
    recorder = Recorder(
        dataset,
        n_samples,
        save_dir,
        max_generated=max_sampling,
        start_successes=resume_state.get('start_successes', 0),
        waiting_list=resume_state.get('waiting_list', None),
        keep_full_batches=_option_enabled(pipeline_opt),
        goal_by_item=resume_state.get('goal_by_item', None),
        successes_by_item=resume_state.get('successes_by_item', None),
        next_sample_index_by_item=resume_state.get('next_sample_index_by_item', None),
        sample_index_stride=resume_state.get('sample_index_stride', 1),
        results_name=resume_state.get('results_name', 'results.jsonl'),
        append=resume_state.get('append_results', False),
    )
    accepted_index = int(resume_state.get('accepted_start_index', 1))
    accepted_index_stride = int(resume_state.get('accepted_index_stride', 1))
    accepted_start = int(recorder.num_successful)
    batches_generated = 0
    stats_path = _generation_stats_path(save_dir, resume_state.get('results_name', 'results.jsonl'))
    tmp_save_dir = os.path.join(save_dir, resume_state.get('tmp_dir_name', 'tmp'))
    n_cycles = 1 if dataset.config.name == 'Molecule' else 0    # at least reconstruct once for small molecules to rectify geometry
    accepted_tmp_dir = _pipeline_tmp_dir(accepted_dir, resume_state)

    prefetch_num_workers = int(prefetch_opt.get('num_workers', 0))
    if max_sampling is not None and prefetch_num_workers > 0:
        print_log(
            'Generation prefetch disabled because max_sampling is set; '
            'this preserves the exact attempt limit.',
            level='WARN',
        )
        prefetch_num_workers = 0
    if _uses_without_replacement_sampler(dataset) and prefetch_num_workers > 1:
        print_log(
            'Generation prefetch workers reduced to 1 because term_rel_pos_sdf '
            'uses sampling without replacement.',
            level='WARN',
        )
        prefetch_num_workers = 1
    if prefetch_num_workers > 0:
        print_log(
            f'Generation prefetch enabled: workers={prefetch_num_workers}, '
            f'prefetch_batches={prefetch_opt.get("prefetch_batches", 2)}, '
            f'backend={prefetch_opt.get("backend", "process")}',
            level='INFO',
        )

    postprocess_num_workers = int(postprocess_opt.get('num_workers', 0))
    if n_cycles != 0 and postprocess_num_workers > 0:
        print_log(
            'Generation postprocess workers disabled for Molecule templates because '
            'autoencoder refinement cycles are sequential.',
            level='WARN',
        )
        postprocess_num_workers = 0
    if postprocess_num_workers > 0:
        if _option_enabled(pipeline_opt):
            accepted_tmp_dir.mkdir(parents=True, exist_ok=True)
        print_log(
            f'Generation postprocess enabled: workers={postprocess_num_workers}, '
            f'backend={postprocess_opt.get("backend", "process")}',
            level='INFO',
        )
    postprocess_run_opt = dict(postprocess_opt)
    postprocess_run_opt['num_workers'] = postprocess_num_workers
    postprocess_executor = _make_postprocess_executor(postprocess_run_opt)

    try:
        with BatchPrefetcher(
            dataset=dataset,
            recorder=recorder,
            batch_size=batch_size,
            num_workers=prefetch_num_workers,
            prefetch_batches=prefetch_opt.get('prefetch_batches', 2),
            backend=prefetch_opt.get('backend', 'process'),
            start_method=prefetch_opt.get('start_method', 'spawn'),
            seed=prefetch_seed,
            torch_num_threads=prefetch_opt.get('torch_num_threads', 1),
        ) as batch_prefetcher:
            while not recorder.is_finished():
                try:
                    batch_list, batch = batch_prefetcher.next()
                except StopIteration:
                    break

                with torch.no_grad():
                    cplx_descs = batch.pop('cplx_desc')
                    batch = to_device(batch, device)
                    batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model.sample(sample_opt=sample_opt, **batch)
                    batches_generated += 1

                if n_cycles == 0:
                    tasks = []
                    for i in range(len(cplx_descs)):
                        item_idx, n = batch_list[i]
                        cplx_desc = cplx_descs[i]
                        tmp_target_dir = os.path.join(tmp_save_dir, cplx_desc.id)
                        target_dir = os.path.join(save_dir, cplx_desc.id)
                        pocket_path = os.path.join(tmp_target_dir, 'pocket.pdb')
                        out_path = os.path.join(target_dir, str(n) + '.pdb')
                        os.makedirs(tmp_target_dir, exist_ok=True)
                        os.makedirs(target_dir, exist_ok=True)
                        if n == 0 or not os.path.exists(pocket_path):
                            complex_to_pdb(cplx_desc.cplx, pocket_path, cplx_desc.tgt_chains)

                        tasks.append({
                            'item_idx': item_idx,
                            'n': n,
                            'cplx_desc': cplx_desc,
                            'template_config': dataset.config,
                            'S': batch_S[i],
                            'X': batch_X[i],
                            'A': batch_A[i],
                            'll': batch_ll[i],
                            'inter_bonds': batch_inter_bonds[i],
                            'intra_bonds': batch_intra_bonds[i],
                            'tmp_target_dir': tmp_target_dir,
                            'target_dir': target_dir,
                            'pocket_path': pocket_path,
                            'out_path': out_path,
                            'check_validity': dataset.config.name == 'Molecule',
                            'check_filters': True,
                            'final_cycle': True,
                            'amide_check_opt': amide_check_opt,
                            'pipeline_opt': pipeline_opt,
                            'pipeline_enabled': _option_enabled(pipeline_opt) and recorder.accepts_more(item_idx),
                            'accepted_tmp_dir': str(accepted_tmp_dir),
                        })

                    for result in _run_postprocess_tasks(tasks, postprocess_executor):
                        item_idx, n, log = result['item_idx'], result['n'], result['log']
                        if _option_enabled(pipeline_opt):
                            if log is not None and recorder.accepts_more(item_idx):
                                try:
                                    log = _finalize_pipeline_temp(log, accepted_dir, accepted_index)
                                except Exception as exc:
                                    print_log(f'Accepted pipeline finalize failed for item={item_idx}, n={n}: {exc}', level='WARN')
                                    _cleanup_pipeline_temp(log)
                                    log = None
                                else:
                                    accepted_index += accepted_index_stride
                            elif log is not None:
                                _cleanup_pipeline_temp(log)
                                log = None
                        recorder.check_and_save(log, item_idx, n)

                    _report_generation_progress(stats_path, recorder, batches_generated, accepted_start)
                    continue

                vae_batch_list = []
                for i in range(len(cplx_descs)):
                    S, X, A, ll = batch_S[i], batch_X[i], batch_A[i], batch_ll[i]
                    inter_bonds, intra_bonds = batch_inter_bonds[i], batch_intra_bonds[i]
                    item_idx, n = batch_list[i]
                    cplx_desc = cplx_descs[i]

                    tmp_target_dir = os.path.join(tmp_save_dir, cplx_desc.id)
                    target_dir = os.path.join(save_dir, cplx_desc.id)
                    pocket_path = os.path.join(tmp_target_dir, 'pocket.pdb')
                    os.makedirs(tmp_target_dir, exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)
                    if n == 0 or not os.path.exists(pocket_path):
                        complex_to_pdb(cplx_desc.cplx, pocket_path, cplx_desc.tgt_chains)

                    out_path = os.path.join(tmp_save_dir, cplx_desc.id, str(n) + '.pdb')
                    cplx_desc, details = data_to_cplx(
                        cplx_desc, dataset.config, S, X, A, ll, inter_bonds, intra_bonds, out_path,
                        check_validity=False,
                        check_filters=False,
                    )
                    if cplx_desc is None:
                        log = None
                        assert n_cycles == 0
                    else:
                        log = None if details is None else format_log(cplx_desc, details, n)

                    if log is None:
                        recorder.check_and_save(log, item_idx, n)
                    else:
                        data = _get_item(
                            os.path.join(tmp_save_dir, cplx_desc.id, f'pocket.pdb'),
                            out_path.rstrip('.pdb') + '.sdf',
                            cplx_desc.tgt_chains
                        )
                        data['cplx_desc'] = cplx_desc
                        vae_batch_list.append(data)

                model_autoencoder = model.autoencoder
                for cyc_i in range(n_cycles):
                    print_log(f'Cycle: {cyc_i}', level='DEBUG')
                    final_cycle = cyc_i == n_cycles - 1
                    batch = dataset.collate_fn(vae_batch_list)
                    vae_batch_list = []
                    with torch.no_grad():
                        if final_cycle: batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
                        cplx_descs = batch.pop('cplx_desc')
                        batch = to_device(batch, device)
                        batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model_autoencoder.generate(**batch)
                    for i in range(len(cplx_descs)):
                        S, X, A, ll = batch_S[i], batch_X[i], batch_A[i], batch_ll[i]
                        inter_bonds, intra_bonds = batch_inter_bonds[i], batch_intra_bonds[i]
                        item_idx, n = batch_list[i]
                        cplx_desc = cplx_descs[i]

                        if final_cycle: out_path = os.path.join(save_dir, cplx_desc.id, str(n) + '.pdb')
                        else: out_path = os.path.join(tmp_save_dir, cplx_desc.id, f'{n}_cyc{cyc_i}.pdb')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        cplx_desc, details = data_to_cplx(
                            cplx_desc, dataset.config, S, X, A, ll, inter_bonds, intra_bonds, out_path,
                            check_validity=dataset.config.name == 'Molecule' and final_cycle,
                            check_filters=final_cycle,
                        )
                        if cplx_desc is None:
                            log = None
                            assert final_cycle
                        else:
                            if final_cycle:
                                details = reject_implausible_ligand_amides(cplx_desc, details, n, amide_check_opt)
                            if details is not None and final_cycle and _option_enabled(pipeline_opt):
                                try:
                                    if recorder.accepts_more(item_idx):
                                        details = run_terminal_merge_pipeline(
                                            cplx_desc,
                                            details,
                                            n,
                                            out_path,
                                            out_path.rstrip('.pdb') + '.sdf',
                                            accepted_dir,
                                            pipeline_opt,
                                            accepted_index,
                                        )
                                    else:
                                        details = None
                                except Exception as exc:
                                    details = _pipeline_reject(f'pipeline error: {exc}', cplx_desc, n)
                                if details is not None:
                                    accepted_index += accepted_index_stride
                            elif details is not None and final_cycle:
                                details = add_terminal_rmsd_details(
                                    cplx_desc,
                                    details,
                                    out_path.rstrip('.pdb') + '.sdf',
                                    pipeline_opt,
                                )
                            log = None if details is None else format_log(cplx_desc, details, n)

                        if final_cycle: recorder.check_and_save(log, item_idx, n)
                        else:
                            data = _get_item(
                                os.path.join(tmp_save_dir, cplx_desc.id, f'pocket.pdb'),
                                out_path.rstrip('.pdb') + '.sdf',
                                cplx_desc.tgt_chains
                            )
                            data['cplx_desc'] = cplx_desc
                            vae_batch_list.append(data)

                _report_generation_progress(stats_path, recorder, batches_generated, accepted_start)
    finally:
        if postprocess_executor is not None:
            postprocess_executor.shutdown(wait=True, cancel_futures=True)

    if recorder.num_generated:
        print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')
    if os.path.exists(tmp_save_dir):
        shutil.rmtree(tmp_save_dir)
    if accepted_tmp_dir.exists():
        shutil.rmtree(accepted_tmp_dir)


def _template_run_name(template, template_idx, class_counts):
    run_name = template.pop('name', None)
    if run_name is not None:
        return str(run_name)

    class_name = template['class']
    if class_counts[class_name] == 1:
        return class_name
    return f'{template_idx:02d}_{class_name}'


def _template_construct_config(template):
    template_config = dict(template)
    for key in (
            'name',
            'pipeline',
            'n_samples',
            'max_sampling',
            'amide_bond_check',
            'sanitize_generation',
            'santinize_generation',
            'sanitize',
            'santinize',
            'resume_generation',
        ):
        template_config.pop(key, None)
    return template_config


def _merge_pipeline_options(config, template):
    pipeline_opt = dict(config.get('pipeline', {}) or {})
    pipeline_opt.update(dict(template.pop('pipeline', {}) or {}))
    return pipeline_opt


def _resolve_goal_samples(config, template, pipeline_opt):
    n_samples = int(template.pop('n_samples', config.get('n_samples', 0)))
    goal = _get_first(pipeline_opt, ('goal_num', 'goal_number', 'target_num'), None)
    if goal is not None:
        n_samples = int(goal)
    if n_samples <= 0:
        raise ValueError('Set n_samples globally or inside each template to a positive integer')
    return n_samples


def _resolve_max_sampling(max_sampling, pipeline_opt, n_samples, dataset_size):
    pipeline_max_sampling = _get_first(
        pipeline_opt,
        ('max_sampling', 'max_sampling_number', 'max_samples', 'max_attempts'),
        None,
    )
    if pipeline_max_sampling is not None:
        max_sampling = pipeline_max_sampling
    if max_sampling is None:
        return None
    max_sampling = int(max_sampling)
    if max_sampling <= 0:
        raise ValueError('max_sampling must be a positive integer')
    goal_attempts = n_samples * dataset_size
    if max_sampling < goal_attempts:
        raise ValueError(
            f'max_sampling={max_sampling} is smaller than the requested goal count '
            f'{goal_attempts} ({n_samples} per input x {dataset_size} input(s))'
        )
    return max_sampling


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    split_opt = _resolve_split_options(config, args)
    base_seed = int(args.seed if args.seed is not None else config.get('seed', 12))
    seed = base_seed + split_opt['index'] if split_opt['enabled'] else base_seed
    setup_seed(seed)
    if split_opt['enabled']:
        print_log(f'Using random seed {seed} (base {base_seed} + shard {split_opt["index"]})')
    else:
        print_log(f'Using random seed {seed}')
    if split_opt['enabled']:
        print_log(
            f'Using generation split shard {split_opt["index"]}/{split_opt["total"]}',
            level='INFO',
        )

    # load model
    ckpt = args.ckpt
    print_log(f'Using checkpoint {ckpt}')
    model = torch_load_full(ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    sample_opt = config.get('sample_opt', {})
    prefetch_opt = _resolve_prefetch_options(config, args)
    postprocess_opt = _resolve_postprocess_options(config, args)
    
    # load dataset and dataloader
    batch_size = config.get('batch_size', 32)
    templates = config['templates']
    class_counts = {}
    for template in templates:
        class_counts[template['class']] = class_counts.get(template['class'], 0) + 1

    for template_idx, template in enumerate(templates):
        template = dict(template)
        pipeline_opt = _merge_pipeline_options(config, template)
        amide_check_opt = _merge_amide_check_options(config, template)
        n_samples = _resolve_goal_samples(config, template, pipeline_opt)
        run_name = _template_run_name(template, template_idx, class_counts)
        max_sampling_config = template.pop('max_sampling', config.get('max_sampling', None))
        print_log(f'Generating for template: {template}')
        print_log(f'Run name: {run_name}, n_samples: {n_samples}')
        run_dir = os.path.join(args.save_dir, run_name)
        out_dir = os.path.join(run_dir, 'candidates')
        os.makedirs(out_dir, exist_ok=True)
        dataset = PDBDataset(**config['dataset'], template_config=R.construct(_template_construct_config(template)))
        n_inputs = len(dataset.pdb_paths)
        max_sampling = _resolve_max_sampling(max_sampling_config, pipeline_opt, n_samples, n_inputs)
        resume_state = None
        if _option_enabled(pipeline_opt):
            if getattr(dataset.config, 'term_rel_pos_sampler', None) is None:
                raise ValueError('pipeline.enabled requires a peptide template with term_rel_pos_sdf')
            resume_opt = _resume_options(config, pipeline_opt)
            sanitize_opt = _sanitize_options(config, pipeline_opt, amide_check_opt)
            if sanitize_opt.get('enabled', False):
                if not resume_opt.get('enabled', False):
                    raise ValueError('sanitize_generation requires resume_generation.enabled=true')
                previous_root = _resume_previous_dir(resume_opt, args.save_dir)
                sanitize_pipeline_accepted_dir(
                    _resume_accepted_dir(previous_root, run_name, pipeline_opt),
                    _dataset_lig_chain_ids(dataset),
                    sanitize_opt,
                )
            resume_state = build_pipeline_resume_state(
                resume_opt,
                args.save_dir,
                run_name,
                out_dir,
                pipeline_opt,
                dataset,
                n_samples,
            )
            resume_state = apply_generation_shard(resume_state, split_opt, dataset, n_samples, run_name)
            max_sampling_for_run = _shard_max_sampling(max_sampling, split_opt)
            print_log(
                f'Accepted pipeline enabled: goal={n_samples * n_inputs}, '
                f'max_sampling={max_sampling_for_run if max_sampling_for_run is not None else "unlimited"}',
                level='INFO'
            )
            if resume_state is not None and resume_state['remaining'] == 0:
                if split_opt['enabled']:
                    print_log(f'Generation shard for {run_name}: no assigned samples, skipping', level='INFO')
                else:
                    print_log(f'Resume generation for {run_name}: target already complete, skipping', level='INFO')
                continue
        else:
            resume_state = apply_generation_shard(None, split_opt, dataset, n_samples, run_name)
            max_sampling_for_run = _shard_max_sampling(max_sampling, split_opt)
            if resume_state is not None and resume_state['remaining'] == 0:
                print_log(f'Generation shard for {run_name}: no assigned samples, skipping', level='INFO')
                continue
        generate_for_one_template(
            model,
            dataset,
            n_samples,
            batch_size,
            out_dir,
            device,
            sample_opt,
            pipeline_opt=pipeline_opt,
            max_sampling=max_sampling_for_run,
            resume_state=resume_state,
            amide_check_opt=amide_check_opt,
            prefetch_opt=prefetch_opt,
            prefetch_seed=seed + template_idx * 10000019,
            postprocess_opt=postprocess_opt,
        )


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    main(args, opt_args)
