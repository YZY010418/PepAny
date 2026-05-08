"""Candidate post-processing helpers for generation."""

import os
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import torch

from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from utils.logger import print_log

from .generation_io import data_to_cplx, format_log
from .generation_pipeline import (
    add_terminal_rmsd_details,
    append_pipeline_metadata,
    reject_implausible_ligand_amides,
    run_terminal_merge_pipeline,
    _pipeline_reject,
)


def _init_postprocess_worker(torch_num_threads):
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))


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
