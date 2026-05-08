"""Generation sharding, option resolution, and progress statistics."""

import json
import os
from pathlib import Path

from utils.logger import print_log

from .generation_common import _get_first
from .generation_resume import _resume_remaining_sampling_budget


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
        'generated_by_item': [0 for _ in range(len(dataset))],
        'accepted_existing_by_item': [0 for _ in range(len(dataset))],
        'candidate_slots_by_item': [[] for _ in range(len(dataset))],
        'waiting_list': [
            (item_idx, n)
            for n in range(n_samples)
            for item_idx in range(len(dataset))
        ],
        'next_sample_index_by_item': [n_samples for _ in range(len(dataset))],
        'start_successes': 0,
        'start_generated': 0,
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

    previous_slots_by_item = state.get('candidate_slots_by_item', [[] for _ in range(len(dataset))])
    sharded_generated_by_item = [
        sum(1 for slot in slots if int(slot) % total == index)
        for slots in previous_slots_by_item
    ]
    previous_accepted_by_item = state.get('accepted_existing_by_item', [0 for _ in range(len(dataset))])
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
        'generated_by_item': sharded_generated_by_item,
        'accepted_existing_by_item': previous_accepted_by_item,
        'waiting_list': sharded_waiting_list,
        'next_sample_index_by_item': [
            int(value) + index
            for value in base_next_sample_index
        ],
        'sample_index_stride': total,
        'results_name': f'results.shard{index}of{total}.jsonl',
        'tmp_dir_name': f'tmp_shard{index}of{total}',
        'start_successes': 0,
        'start_generated': sum(sharded_generated_by_item),
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


def _apply_remaining_sampling_budget(max_sampling, resume_state, resume_opt, split_opt, run_name):
    remaining_budget = _resume_remaining_sampling_budget(resume_opt)
    if remaining_budget is None:
        return max_sampling
    start_generated = int((resume_state or {}).get('start_generated', 0))
    max_sampling = start_generated + remaining_budget
    shard_note = ' for this shard' if split_opt.get('enabled', False) else ''
    print_log(
        f'Resume generation for {run_name}: remaining_sampling_budget={remaining_budget}'
        f'{shard_note} overrides max_sampling_number; '
        f'previous_candidates={start_generated}, effective_max_sampling={max_sampling}',
        level='INFO',
    )
    return max_sampling


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


def _target_generation_stats(
        recorder,
        target_ids,
        generated_start_by_item,
        accepted_start_by_item,
        accepted_existing_by_item,
    ):
    if target_ids is None:
        return {}
    per_target = {}
    for item_idx, target_id in enumerate(target_ids):
        generated_total = int(recorder.generated_by_item[item_idx])
        generated_start = int(generated_start_by_item[item_idx])
        accepted_control = int(recorder.successes_by_item[item_idx])
        accepted_existing = int(accepted_existing_by_item[item_idx])
        accepted_total = max(accepted_existing, accepted_control)
        accepted_start = max(accepted_existing, int(accepted_start_by_item[item_idx]))
        accepted_this_run = max(0, accepted_total - accepted_start)
        generated_this_run = max(0, generated_total - generated_start)
        per_target[target_id] = {
            'generated_candidates': generated_total,
            'generated_candidates_this_run': generated_this_run,
            'accepted': accepted_total,
            'accepted_this_run': accepted_this_run,
            'goal': int(recorder.goal_by_item[item_idx]),
            'remaining': max(0, int(recorder.goal_by_item[item_idx]) - accepted_control),
            'success_rate': accepted_total / generated_total if generated_total else 0.0,
            'success_rate_this_run': accepted_this_run / generated_this_run if generated_this_run else 0.0,
        }
    return per_target


def _write_generation_stats(
        stats_path,
        recorder,
        batches_generated,
        accepted_start,
        generated_start=0,
        target_ids=None,
        generated_start_by_item=None,
        accepted_start_by_item=None,
        accepted_existing_by_item=None,
    ):
    generated = int(recorder.num_generated)
    generated_this_run = max(0, generated - int(generated_start))
    accepted = int(recorder.num_successful - accepted_start)
    if generated_start_by_item is None:
        generated_start_by_item = [0 for _ in range(recorder.num_items)]
    if accepted_start_by_item is None:
        accepted_start_by_item = [0 for _ in range(recorder.num_items)]
    if accepted_existing_by_item is None:
        accepted_existing_by_item = [0 for _ in range(recorder.num_items)]
    per_target = _target_generation_stats(
        recorder,
        target_ids,
        generated_start_by_item,
        accepted_start_by_item,
        accepted_existing_by_item,
    )
    accepted_total = (
        sum(item['accepted'] for item in per_target.values())
        if per_target else int(recorder.num_successful)
    )
    stats = {
        'batches_generated': int(batches_generated),
        'generated_candidates': generated,
        'generated_candidates_this_run': generated_this_run,
        'failed_candidates': int(recorder.num_failed),
        'accepted': accepted,
        'accepted_total': int(accepted_total),
        'target_total': int(recorder.total),
        'accept_rate': accepted_total / generated if generated else 0.0,
        'accept_rate_this_run': accepted / generated_this_run if generated_this_run else 0.0,
        'per_target': per_target,
    }
    stats_path = Path(stats_path)
    tmp_path = stats_path.with_suffix(stats_path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(stats, indent=2) + '\n')
    os.replace(tmp_path, stats_path)
    return stats


def _report_generation_progress(
        stats_path,
        recorder,
        batches_generated,
        accepted_start,
        generated_start=0,
        target_ids=None,
        generated_start_by_item=None,
        accepted_start_by_item=None,
        accepted_existing_by_item=None,
    ):
    stats = _write_generation_stats(
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
    message = (
        f'{stats["batches_generated"]} batches generated '
        f'({stats["generated_candidates_this_run"]} designs this run, '
        f'{stats["generated_candidates"]} designs in total), '
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
