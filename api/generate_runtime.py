#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil

import yaml
import torch

from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from utils.config_utils import overwrite_values
from utils.io import torch_load_full
from utils.logger import print_log
from utils.random_seed import setup_seed
import utils.register as R
from models.LDM.data_utils import _get_item, Recorder

from .pdb_dataset import PDBDataset
from . import generation_common as _generation_common
from . import generation_io as _generation_io
from . import generation_pipeline as _generation_pipeline
from . import generation_postprocess as _generation_postprocess
from . import generation_prefetch as _generation_prefetch
from . import generation_progress as _generation_progress
from . import generation_resume as _generation_resume
from . import generation_sanitize as _generation_sanitize

from .generation_cli import parse
from .generation_common import (
    _get_first,
    _option_enabled,
    _template_uses_capping,
    to_device,
)
from .generation_io import data_to_cplx, format_log
from .generation_prefetch import BatchPrefetcher
from .generation_pipeline import (
    _merge_amide_check_options,
    _pipeline_reject,
    add_terminal_rmsd_details,
    reject_implausible_ligand_amides,
    resolve_pipeline_accepted_dir,
    run_terminal_merge_pipeline,
)
from .generation_postprocess import (
    _cleanup_pipeline_temp,
    _finalize_pipeline_temp,
    _make_postprocess_executor,
    _pipeline_tmp_dir,
    _run_postprocess_tasks,
)
from .generation_progress import (
    _apply_remaining_sampling_budget,
    _generation_stats_path,
    _report_generation_progress,
    _resolve_postprocess_options,
    _resolve_prefetch_options,
    _resolve_split_options,
    _shard_max_sampling,
    _uses_without_replacement_sampler,
    apply_generation_shard,
)
from .generation_resume import (
    _dataset_item_target_ids,
    _dataset_lig_chain_ids,
    _resume_accepted_dir,
    _resume_options,
    _resume_previous_dir,
    build_pipeline_resume_state,
)
from .generation_sanitize import (
    _sanitize_options,
    sanitize_pipeline_accepted_dir,
)


_HELPER_MODULES = (
    _generation_common,
    _generation_io,
    _generation_prefetch,
    _generation_pipeline,
    _generation_postprocess,
    _generation_resume,
    _generation_sanitize,
    _generation_progress,
)


def __getattr__(name):
    for module in _HELPER_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


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
        start_generated=resume_state.get('start_generated', 0),
        generated_by_item=resume_state.get('generated_by_item', None),
    )
    accepted_index = int(resume_state.get('accepted_start_index', 1))
    accepted_index_stride = int(resume_state.get('accepted_index_stride', 1))
    accepted_start = int(recorder.num_successful)
    generated_start = int(recorder.num_generated)
    target_ids = _dataset_item_target_ids(dataset)
    accepted_start_by_item = list(recorder.successes_by_item)
    generated_start_by_item = list(recorder.generated_by_item)
    accepted_existing_by_item = list(resume_state.get(
        'accepted_existing_by_item',
        accepted_start_by_item,
    ))
    batches_generated = 0
    if max_sampling is not None and generated_start:
        remaining_attempt_budget = max(0, int(max_sampling) - generated_start)
        print_log(
            f'Resume max_sampling accounting: previous_candidates={generated_start}, '
            f'max_sampling={int(max_sampling)}, remaining_attempt_budget={remaining_attempt_budget}',
            level='INFO',
        )
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

                    _report_generation_progress(
                        stats_path,
                        recorder,
                        batches_generated,
                        accepted_start,
                        generated_start=generated_start,
                        target_ids=target_ids,
                        generated_start_by_item=generated_start_by_item,
                        accepted_start_by_item=accepted_start_by_item,
                        accepted_existing_by_item=accepted_existing_by_item,
                    )
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

                _report_generation_progress(
                    stats_path,
                    recorder,
                    batches_generated,
                    accepted_start,
                    generated_start=generated_start,
                    target_ids=target_ids,
                    generated_start_by_item=generated_start_by_item,
                    accepted_start_by_item=accepted_start_by_item,
                    accepted_existing_by_item=accepted_existing_by_item,
                )
    finally:
        if postprocess_executor is not None:
            postprocess_executor.shutdown(wait=True, cancel_futures=True)

    if batches_generated == 0:
        _report_generation_progress(
            stats_path,
            recorder,
            batches_generated,
            accepted_start,
            generated_start=generated_start,
            target_ids=target_ids,
            generated_start_by_item=generated_start_by_item,
            accepted_start_by_item=accepted_start_by_item,
            accepted_existing_by_item=accepted_existing_by_item,
        )

    current_generated = recorder.num_generated - generated_start
    if current_generated:
        print_log(f'Failed rate: {recorder.num_failed / current_generated}', level='DEBUG')
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
        run_name = _template_run_name(template, template_idx, class_counts)
        if _option_enabled(pipeline_opt) and not _template_uses_capping(template):
            print_log(
                f'{run_name}: capping=false disables the accepted cyclic pipeline; '
                'terminal cap merging requires the synthetic ACE/NME capping scaffold.',
                level='WARN',
            )
            pipeline_opt['enabled'] = False
            pipeline_opt['merge'] = False
        amide_check_opt = _merge_amide_check_options(config, template)
        n_samples = _resolve_goal_samples(config, template, pipeline_opt)
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
            max_sampling_for_run = _apply_remaining_sampling_budget(
                max_sampling_for_run,
                resume_state,
                resume_opt,
                split_opt,
                run_name,
            )
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
