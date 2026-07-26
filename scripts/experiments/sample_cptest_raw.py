#!/usr/bin/env python3
"""Run the matched-budget, filter-free CPtest sampling experiment."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate an exact number of raw PepAny attempts per target and cyclization '
            'type without rejection sampling or post-generation filters.'
        )
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Base generation YAML with CPtest and template definitions',
    )
    parser.add_argument('--ckpt', default='checkpoints/pepany_v1.ckpt', help='PepAny checkpoint')
    parser.add_argument('--save-dir', required=True, help='New, empty output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index; use -1 for CPU')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--samples-per-target', type=int, default=100)
    parser.add_argument('--guidance-scale', type=float, default=2.5)
    parser.add_argument('--pocket-threshold', type=float, default=15.0)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-shards', type=int, default=None)
    parser.add_argument('--shard-id', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--postprocess-workers', type=int, default=None)
    return parser.parse_args()


def controlled_config(base_config, args):
    if not isinstance(base_config, dict):
        raise ValueError('The base config must contain a YAML mapping')
    if 'dataset' not in base_config or 'templates' not in base_config:
        raise ValueError('The base config must define dataset and templates')

    config = dict(base_config)
    config['dataset'] = dict(config['dataset'])
    if not config['dataset'].get('lig_chains'):
        raise ValueError(
            'A reference ligand chain is required to define the controlled 15 Angstrom pocket'
        )
    manual_pocket_keys = (
        'pocket_residues',
        'pocket_residue_tsv',
        'pocket_residues_tsv',
    )
    if any(config['dataset'].get(key) is not None for key in manual_pocket_keys):
        raise ValueError(
            'Remove manual pocket residue settings for this experiment; the pocket must be '
            'defined by --pocket-threshold around the reference ligand'
        )
    config['dataset']['pocket_threshold'] = args.pocket_threshold
    config['sample_opt'] = dict(config.get('sample_opt', {}))
    config['sample_opt']['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    config['n_samples'] = args.samples_per_target
    config['raw_sampling'] = {
        'enabled': True,
        'copy_conditioning_conformer': True,
    }
    config['pipeline'] = {'enabled': False}
    config['amide_bond_check'] = {'enabled': False}
    config['resume_generation'] = {'enabled': False}
    config['sanitize_generation'] = {'enabled': False}
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    templates = []
    for base_template in config['templates']:
        template = dict(base_template)
        if template.get('class') != 'Peptide':
            raise ValueError(
                f'Raw CPtest sampling requires Peptide templates, got {template.get("class")!r}'
            )
        if 'term_rel_pos_sdf' not in template:
            raise ValueError(
                f'Cyclization template {template.get("name", "<unnamed>")!r} must define '
                'term_rel_pos_sdf so its conditioning conformer can be preserved'
            )
        template['n_samples'] = args.samples_per_target
        template.pop('max_sampling', None)
        template['pipeline'] = {'enabled': False}
        template['amide_bond_check'] = {'enabled': False}
        template['raw_sampling'] = {
            'enabled': True,
            'copy_conditioning_conformer': True,
        }
        templates.append(template)
    config['templates'] = templates
    config.pop('max_sampling', None)
    return config


def validate_args(args):
    if args.samples_per_target <= 0:
        raise ValueError('--samples-per-target must be positive')
    if args.pocket_threshold <= 0:
        raise ValueError('--pocket-threshold must be positive')
    if args.num_shards is not None:
        if args.num_shards <= 0:
            raise ValueError('--num-shards must be positive')
        if args.shard_id is None:
            raise ValueError('--shard-id is required when --num-shards is set')
    if args.shard_id is not None and args.num_shards is None:
        raise ValueError('--num-shards is required when --shard-id is set')


def prepare_output_directory(save_dir, config, args, yaml):
    save_dir = Path(save_dir)
    if args.num_shards is None:
        if save_dir.exists() and any(save_dir.iterdir()):
            raise FileExistsError(
                f'Output directory is not empty: {save_dir}. Use a new directory so the '
                'controlled attempt count cannot be mixed with a previous run.'
            )
        effective_config = save_dir / 'effective_config.yaml'
    else:
        effective_config = save_dir / (
            f'effective_config.shard{args.shard_id}of{args.num_shards}.yaml'
        )
        marker = save_dir / f'.shard{args.shard_id}of{args.num_shards}.started'
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            marker.touch(exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f'Shard {args.shard_id}/{args.num_shards} has already been started in '
                f'{save_dir}; use a new directory to avoid overwriting attempts.'
            ) from exc

    save_dir.mkdir(parents=True, exist_ok=True)
    effective_config.write_text(yaml.safe_dump(config, sort_keys=False))
    return save_dir, effective_config


def main():
    args = parse_args()
    validate_args(args)
    import yaml

    from api import generate_runtime

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f'Config does not exist: {config_path}')
    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')

    with config_path.open() as handle:
        config = controlled_config(yaml.safe_load(handle), args)
    save_dir, effective_config = prepare_output_directory(
        args.save_dir,
        config,
        args,
        yaml,
    )

    runtime_args = SimpleNamespace(
        config=str(effective_config),
        ckpt=str(checkpoint_path),
        save_dir=str(save_dir),
        gpu=args.gpu,
        seed=args.seed,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        postprocess_workers=args.postprocess_workers,
        n_cpus=4,
    )
    generate_runtime.main(runtime_args, [])


if __name__ == '__main__':
    main()
