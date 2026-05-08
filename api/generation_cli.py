"""Command-line parsing for the generation entry points."""

import argparse


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
