"""Sanitizing helpers for accepted generation outputs."""

import csv
import shutil
try:
    import fcntl
except ImportError:
    fcntl = None
from pathlib import Path

from data.bioparse.parser.cif_to_complex import cif_to_complex
from utils.logger import print_log

from .generation_pipeline import (
    PIPELINE_METADATA_FIELDS,
    _accepted_index,
    _format_ligand_amide_violation,
    _read_pipeline_metadata,
    find_ligand_amide_bond_violations,
)


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
