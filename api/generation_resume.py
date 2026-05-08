"""Resume-state helpers for generation."""

import json
import os
from pathlib import Path

from utils.logger import print_log

from .generation_common import _get_first, _safe_int
from .generation_pipeline import (
    _accepted_index,
    _pipeline_output_dir_name,
    _read_pipeline_metadata,
)


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


def _candidate_slots(candidate_dirs, target_id):
    if isinstance(candidate_dirs, (str, Path)):
        candidate_dirs = [candidate_dirs]
    slots = set()
    for candidate_dir in candidate_dirs:
        target_dir = Path(candidate_dir) / target_id
        if not target_dir.exists():
            continue
        for candidate_path in list(target_dir.glob('*.pdb')) + list(target_dir.glob('*.sdf')):
            slot = _safe_int(candidate_path.stem)
            if slot is not None:
                slots.add(slot)
    return slots


def _max_candidate_slot(candidate_dirs, target_id):
    slots = _candidate_slots(candidate_dirs, target_id)
    return max(slots) if slots else -1


def _candidate_attempt_count(candidate_dirs, target_id):
    return len(_candidate_slots(candidate_dirs, target_id))


def _candidate_stats_files(candidate_dir):
    candidate_dir = Path(candidate_dir)
    exact_path = candidate_dir / 'generation_stats.json'
    if exact_path.exists():
        return [exact_path]
    return sorted(candidate_dir.glob('generation_stats*.json'))


def _read_generation_stats_file(stats_path):
    try:
        with Path(stats_path).open() as handle:
            data = json.load(handle)
    except Exception as exc:
        print_log(f'Could not read generation stats {stats_path}: {exc}', level='WARN')
        return None
    if not isinstance(data, dict):
        print_log(f'Ignoring malformed generation stats {stats_path}: expected JSON object', level='WARN')
        return None
    return data


def _read_generation_stats(candidate_dirs):
    if isinstance(candidate_dirs, (str, Path)):
        candidate_dirs = [candidate_dirs]
    stats_files = []
    seen = set()
    for candidate_dir in candidate_dirs:
        for stats_path in _candidate_stats_files(candidate_dir):
            key = str(stats_path.resolve()) if stats_path.exists() else str(stats_path.absolute())
            if key in seen:
                continue
            seen.add(key)
            stats_files.append(stats_path)

    if not stats_files:
        return None

    total_generated = 0
    has_total = False
    per_target_generated = {}
    for stats_path in stats_files:
        stats = _read_generation_stats_file(stats_path)
        if stats is None:
            continue
        generated = _safe_int(stats.get('generated_candidates', None))
        if generated is not None:
            total_generated += generated
            has_total = True
        per_target = stats.get('per_target', {})
        if isinstance(per_target, dict):
            for target_id, target_stats in per_target.items():
                if not isinstance(target_stats, dict):
                    continue
                target_generated = _safe_int(target_stats.get('generated_candidates', None))
                if target_generated is not None:
                    per_target_generated[target_id] = per_target_generated.get(target_id, 0) + target_generated

    if not has_total and not per_target_generated:
        return None
    return {
        'stats_files': [str(path) for path in stats_files],
        'generated_candidates': total_generated if has_total else None,
        'per_target_generated': per_target_generated,
    }


def _resume_candidate_dir(previous_dir, run_name):
    root = Path(previous_dir)
    candidates = [
        root / run_name / 'candidates',
        root / 'candidates',
        root,
    ]
    for candidate in candidates:
        if (
                any(candidate.glob('generation_stats*.json'))
                or any(candidate.glob('*/*.pdb'))
                or any(candidate.glob('*/*.sdf'))
            ):
            return candidate
    return candidates[0]


def _unique_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        path = Path(path)
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


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


def _resume_remaining_sampling_budget(resume_opt):
    raw_budget = _get_first(
        resume_opt,
        (
            'remaining_sampling_budget',
            'remaining_sample_budget',
            'remaining_max_sampling',
            'remaining_max_sampling_number',
        ),
        None,
    )
    if raw_budget is None:
        return None
    budget = int(raw_budget)
    if budget < 0:
        raise ValueError('resume_generation.remaining_sampling_budget must be >= 0')
    return budget


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
    previous_candidate_dir = _resume_candidate_dir(previous_root, run_name)
    candidate_dirs = _unique_paths([previous_candidate_dir, candidate_dir])
    generation_stats = _read_generation_stats(candidate_dirs)
    per_target_generated = {}
    stats_total_generated = None
    if generation_stats is not None:
        per_target_generated = generation_stats.get('per_target_generated', {})
        stats_total_generated = generation_stats.get('generated_candidates', None)
        print_log(
            f'Resume generation stats for {run_name}: '
            f'files={generation_stats.get("stats_files", [])}, '
            f'generated_candidates={stats_total_generated}',
            level='INFO',
        )
    existing = _existing_pipeline_state(accepted_dir, target_ids)
    counts_by_id = existing['counts_by_id']
    sample_slot_max_by_id = existing['sample_slot_max_by_id']

    goal_by_item, successes_by_item, waiting_list, next_sample_index_by_item = [], [], [], []
    generated_by_item, accepted_existing_by_item, candidate_slots_by_item = [], [], []
    total_remaining = 0
    for item_idx, target_id in enumerate(target_ids):
        existing_count = counts_by_id.get(target_id, 0)
        candidate_slots = sorted(_candidate_slots(candidate_dirs, target_id))
        file_attempts = len(candidate_slots)
        stats_attempts = per_target_generated.get(target_id, None)
        existing_attempts = int(stats_attempts) if stats_attempts is not None else file_attempts
        clipped_existing = min(existing_count, n_samples)
        remaining = max(0, n_samples - clipped_existing)
        goal_by_item.append(n_samples)
        successes_by_item.append(clipped_existing)
        generated_by_item.append(existing_attempts)
        accepted_existing_by_item.append(existing_count)
        candidate_slots_by_item.append(candidate_slots)
        next_slot = max(
            n_samples,
            sample_slot_max_by_id.get(target_id, -1) + 1,
            _max_candidate_slot(candidate_dirs, target_id) + 1,
        )
        for offset in range(remaining):
            waiting_list.append((item_idx, next_slot + offset))
        next_sample_index_by_item.append(next_slot + remaining)
        total_remaining += remaining
        success_rate = existing_count / existing_attempts if existing_attempts else 0.0
        attempt_source = 'generation_stats.json' if stats_attempts is not None else 'candidate files'
        print_log(
            f'Resume target {run_name}/{target_id}: '
            f'existing={existing_count}, previous_candidates={existing_attempts} '
            f'({attempt_source}), '
            f'success_rate={success_rate:.4f}, goal={n_samples}, to_generate={remaining}',
            level='INFO',
        )

    start_generated = (
        int(stats_total_generated)
        if stats_total_generated is not None
        else sum(generated_by_item)
    )
    print_log(
        f'Resume generation for {run_name}: previous accepted dir={accepted_dir}, '
        f'previous candidates dir={previous_candidate_dir}, '
        f'previous_candidates={start_generated}, total_to_generate={total_remaining}',
        level='INFO',
    )
    return {
        'accepted_start_index': existing['max_accepted_index'] + 1,
        'append_results': True,
        'goal_by_item': goal_by_item,
        'successes_by_item': successes_by_item,
        'generated_by_item': generated_by_item,
        'accepted_existing_by_item': accepted_existing_by_item,
        'candidate_slots_by_item': candidate_slots_by_item,
        'waiting_list': waiting_list,
        'next_sample_index_by_item': next_sample_index_by_item,
        'start_successes': sum(successes_by_item),
        'start_generated': start_generated,
        'remaining': total_remaining,
        'previous_accepted_dir': str(accepted_dir),
        'previous_candidate_dir': str(previous_candidate_dir),
    }
