"""Persistence helpers for controlled, filter-free generation attempts."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from data.bioparse.writer.complex_to_pdb import complex_to_pdb


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__} to JSON')


def save_raw_attempt(
        target_dir,
        cplx_desc,
        sample_index,
        model_output,
        sample_opt,
        run_name,
        pocket_threshold,
        copy_conditioning_conformer=True,
    ):
    """Save one literal model output and the conformers used to condition it."""
    target_dir = Path(target_dir)
    raw_dir = target_dir / 'raw_model_outputs'
    conformer_dir = target_dir / 'conditioning_conformers'
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / 'target.pdb'
    if not target_path.exists():
        temporary_target_path = target_dir / f'.target.{os.getpid()}.pdb'
        try:
            complex_to_pdb(
                cplx_desc.cplx,
                str(temporary_target_path),
                cplx_desc.tgt_chains,
            )
            os.replace(temporary_target_path, target_path)
        finally:
            temporary_target_path.unlink(missing_ok=True)

    control = dict(cplx_desc.control or {})
    source_conformer = control.get('term_rel_pos_sdf_path')
    saved_conformer = None
    if copy_conditioning_conformer and source_conformer:
        source_path = Path(source_conformer)
        if not source_path.is_file():
            raise FileNotFoundError(f'Conditioning conformer does not exist: {source_path}')
        conformer_dir.mkdir(parents=True, exist_ok=True)
        saved_conformer = conformer_dir / f'{sample_index}.sdf'
        shutil.copy2(source_path, saved_conformer)

    raw_path = raw_dir / f'{sample_index}.pt'
    metadata_path = raw_dir / f'{sample_index}.json'
    candidate_stem = target_dir / str(sample_index)
    metadata = {
        'format_version': 1,
        'run_name': str(run_name),
        'target_id': str(cplx_desc.id),
        'sample_index': int(sample_index),
        'target_conformer': str(target_path.resolve()),
        'conditioning_conformer': (
            str(saved_conformer.resolve()) if saved_conformer is not None else None
        ),
        'conditioning_conformer_source': (
            str(Path(source_conformer).resolve()) if source_conformer else None
        ),
        'raw_model_output': str(raw_path.resolve()),
        'candidate_pdb': str(candidate_stem.with_suffix('.pdb').resolve()),
        'candidate_sdf': str(candidate_stem.with_suffix('.sdf').resolve()),
        'pocket_threshold_angstrom': float(pocket_threshold),
        'sample_opt': dict(sample_opt),
        'control': control,
    }
    payload = {
        'metadata': metadata,
        'model_output': _to_cpu(model_output),
    }
    torch.save(payload, raw_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_value) + '\n'
    )
    return {
        'raw_model_output': str(raw_path.resolve()),
        'raw_metadata': str(metadata_path.resolve()),
        'target_conformer': str(target_path.resolve()),
        'conditioning_conformer': metadata['conditioning_conformer'],
        'conditioning_conformer_source': metadata['conditioning_conformer_source'],
    }
