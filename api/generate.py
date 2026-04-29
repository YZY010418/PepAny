#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil
import argparse
from pathlib import Path

import yaml
import torch
from rdkit import Chem

from data.bioparse.hierarchy import Atom, Block, Bond, Complex, Molecule, merge_cplx, remove_mols
from data.bioparse.utils import bond_type_from_rdkit
from data.bioparse.writer.complex_to_cif import complex_to_cif
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from evaluation.terminal_amide_rmsd import (
    detect_clashes,
    detect_receptor_clashes,
    get_positions,
    infer_generated_indexes,
    kabsch_align,
    load_sdf_mol,
    merge_terminal_cap,
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


def rdkit_mol_to_ligand_complex(mol, chain_id='L', residue_name='CYC'):
    conf = mol.GetConformer()
    atoms = []
    name_counts = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        symbol = atom.GetSymbol()
        atoms.append(Atom(
            name=_atom_name(symbol, name_counts),
            coordinate=[float(pos.x), float(pos.y), float(pos.z)],
            element=symbol,
            id=str(idx + 1),
        ))

    block = Block(
        name=str(residue_name)[:3],
        atoms=atoms,
        id=(1, ''),
        properties={'original_name': str(residue_name)[:3]},
    )
    bonds = []
    for bond in mol.GetBonds():
        bonds.append(Bond(
            index1=(0, 0, bond.GetBeginAtomIdx()),
            index2=(0, 0, bond.GetEndAtomIdx()),
            bond_type=bond_type_from_rdkit(bond),
        ))
    return Complex(
        name='merged_ligand',
        molecules=[Molecule(name=str(residue_name), blocks=[block], id=str(chain_id))],
        bonds=bonds,
    )


def merged_complex_from_mol(cplx_desc, merged_mol, residue_name='CYC'):
    receptor_cplx = remove_mols(cplx_desc.cplx, cplx_desc.lig_chains)
    ligand_cplx = rdkit_mol_to_ligand_complex(
        merged_mol,
        chain_id=cplx_desc.lig_chains[0],
        residue_name=residue_name,
    )
    return merge_cplx(receptor_cplx, ligand_cplx)


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


def _pipeline_reject(reason, cplx_desc, n):
    print_log(f'Rejected {cplx_desc.id}/{n}: {reason}', level='DEBUG')
    return None


def run_terminal_merge_pipeline(cplx_desc, details, n, raw_pdb_path, raw_sdf_path, accepted_dir, pipeline_opt):
    control = cplx_desc.control or {}
    ref_path = control.get('term_rel_pos_sdf_path', None)
    ref_atom_indices = control.get('term_rel_pos_atom_indices', None)
    if ref_path is None or ref_atom_indices is None:
        raise ValueError('pipeline.enabled requires term_rel_pos_sdf so each sample has a reference SDF')

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

    residue_name = pipeline_opt.get('ligand_residue_name', 'CYC')
    accepted_subdir = Path(accepted_dir) / cplx_desc.id
    accepted_subdir.mkdir(parents=True, exist_ok=True)
    ref_stem = _sdf_stem(ref_path)
    cif_path = accepted_subdir / f'{n}_{ref_stem}.cif'
    merged_cplx = merged_complex_from_mol(cplx_desc, merged_mol, residue_name=residue_name)
    complex_to_cif(
        merged_cplx,
        str(cif_path),
        selected_chains=cplx_desc.tgt_chains + cplx_desc.lig_chains,
        title=f'{cplx_desc.id}_{n}_{ref_stem}',
    )

    merged_sdf_path = ''
    if bool(pipeline_opt.get('write_merged_sdf', False)):
        merged_sdf_path = str(accepted_subdir / f'{n}_{ref_stem}.sdf')
        write_sdf(merged_mol, merged_sdf_path)

    details.update({
        'terminal_rmsd': terminal_rmsd,
        'n_term_rmsd': n_term_rmsd,
        'c_term_rmsd': c_term_rmsd,
        'rmsd_cutoff': rmsd_cutoff,
        'self_clash_count': len(self_clashes),
        'receptor_clash_count': len(receptor_clashes),
        'reference_sdf': str(ref_path),
        'raw_pdb': str(raw_pdb_path),
        'raw_sdf': str(raw_sdf_path),
        'accepted_cif': str(cif_path),
        'merged_sdf': merged_sdf_path,
        'merge_status': 'saved',
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
    ):
    pipeline_opt = {} if pipeline_opt is None else dict(pipeline_opt)
    output_dir = pipeline_opt.get('output_dir', None)
    if output_dir is None:
        accepted_dir = os.path.join(os.path.dirname(save_dir), 'accepted')
    elif os.path.isabs(output_dir):
        accepted_dir = output_dir
    else:
        accepted_dir = os.path.join(os.path.dirname(save_dir), output_dir)
    if _option_enabled(pipeline_opt):
        os.makedirs(accepted_dir, exist_ok=True)
    recorder = Recorder(dataset, n_samples, save_dir, max_generated=max_sampling)
    tmp_save_dir = os.path.join(save_dir, 'tmp')
    n_cycles = 1 if dataset.config.name == 'Molecule' else 0    # at least reconstruct once for small molecules to rectify geometry
    
    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = dataset.collate_fn([dataset[i] for i, _ in batch_list])
        
        with torch.no_grad():
            cplx_descs = batch.pop('cplx_desc')
            # to GPU
            batch = to_device(batch, device)
            # inference
            batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model.sample(sample_opt=sample_opt, **batch)

        vae_batch_list = []
        for i in range(len(cplx_descs)):
            S, X, A, ll = batch_S[i], batch_X[i], batch_A[i], batch_ll[i]
            inter_bonds, intra_bonds = batch_inter_bonds[i], batch_intra_bonds[i]
            item_idx, n = batch_list[i]
            cplx_desc = cplx_descs[i]

            if n == 0:
                os.makedirs(os.path.join(tmp_save_dir, cplx_desc.id), exist_ok=True)
                os.makedirs(os.path.join(save_dir, cplx_desc.id), exist_ok=True)
                complex_to_pdb(cplx_desc.cplx, os.path.join(tmp_save_dir, cplx_desc.id, 'pocket.pdb'), cplx_desc.tgt_chains)

            if n_cycles == 0: out_path = os.path.join(save_dir, cplx_desc.id, str(n) + '.pdb')
            else: out_path = os.path.join(tmp_save_dir, cplx_desc.id, str(n) + '.pdb')
            cplx_desc, details = data_to_cplx(
                cplx_desc, dataset.config, S, X, A, ll, inter_bonds, intra_bonds, out_path,
                check_validity = dataset.config.name == 'Molecule' and (n_cycles == 0),
                check_filters = (n_cycles == 0)
            )
            if cplx_desc is None:
                log = None
                assert n_cycles == 0  # only the last cycle has such possibility
            else:
                if n_cycles == 0 and _option_enabled(pipeline_opt):
                    try:
                        details = run_terminal_merge_pipeline(
                            cplx_desc,
                            details,
                            n,
                            out_path,
                            out_path.rstrip('.pdb') + '.sdf',
                            accepted_dir,
                            pipeline_opt,
                        )
                    except Exception as exc:
                        details = _pipeline_reject(f'pipeline error: {exc}', cplx_desc, n)
                log = None if details is None else format_log(cplx_desc, details, n)
            
            if n_cycles == 0: recorder.check_and_save(log, item_idx, n)
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
                    check_validity = dataset.config.name == 'Molecule' and final_cycle,
                    check_filters = final_cycle
                )
                if cplx_desc is None:
                    log = None
                    assert final_cycle  # only the last cycle has such possibility
                else:
                    if final_cycle and _option_enabled(pipeline_opt):
                        try:
                            details = run_terminal_merge_pipeline(
                                cplx_desc,
                                details,
                                n,
                                out_path,
                                out_path.rstrip('.pdb') + '.sdf',
                                accepted_dir,
                                pipeline_opt,
                            )
                        except Exception as exc:
                            details = _pipeline_reject(f'pipeline error: {exc}', cplx_desc, n)
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

    if recorder.num_generated:
        print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')
    if os.path.exists(tmp_save_dir):
        shutil.rmtree(tmp_save_dir)


def _template_run_name(template, template_idx, class_counts):
    run_name = template.pop('name', None)
    if run_name is not None:
        return str(run_name)

    class_name = template['class']
    if class_counts[class_name] == 1:
        return class_name
    return f'{template_idx:02d}_{class_name}'


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


def _resolve_max_sampling(config, template, pipeline_opt, n_samples, dataset_size):
    max_sampling = template.pop('max_sampling', config.get('max_sampling', None))
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

    # load model
    ckpt = args.ckpt
    print_log(f'Using checkpoint {ckpt}')
    model = torch_load_full(ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    sample_opt = config.get('sample_opt', {})
    
    # load dataset and dataloader
    batch_size = config.get('batch_size', 32)
    templates = config['templates']
    class_counts = {}
    for template in templates:
        class_counts[template['class']] = class_counts.get(template['class'], 0) + 1

    for template_idx, template in enumerate(templates):
        template = dict(template)
        pipeline_opt = _merge_pipeline_options(config, template)
        n_samples = _resolve_goal_samples(config, template, pipeline_opt)
        run_name = _template_run_name(template, template_idx, class_counts)
        max_sampling = _resolve_max_sampling(config, template, pipeline_opt, n_samples, len(config['dataset']['pdb_paths']))
        print_log(f'Generating for template: {template}')
        print_log(f'Run name: {run_name}, n_samples: {n_samples}')
        run_dir = os.path.join(args.save_dir, run_name)
        out_dir = os.path.join(run_dir, 'candidates')
        os.makedirs(out_dir, exist_ok=True)
        dataset = PDBDataset(**config['dataset'], template_config=R.construct(template))
        if _option_enabled(pipeline_opt):
            if max_sampling is None:
                raise ValueError('pipeline.enabled requires max_sampling/max_sampling_number')
            if getattr(dataset.config, 'term_rel_pos_sampler', None) is None:
                raise ValueError('pipeline.enabled requires a peptide template with term_rel_pos_sdf')
            print_log(
                f'Accepted pipeline enabled: goal={n_samples * len(dataset)}, '
                f'max_sampling={max_sampling}',
                level='INFO'
            )
        generate_for_one_template(
            model,
            dataset,
            n_samples,
            batch_size,
            out_dir,
            device,
            sample_opt,
            pipeline_opt=pipeline_opt,
            max_sampling=max_sampling,
        )


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)
