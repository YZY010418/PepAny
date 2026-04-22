#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Dynamic CIF Dataset - loads .cif.gz files on-demand without preprocessing.

Features:
- Memory efficient: only loads one file at a time
- No disk space needed for mmap preprocessing
- Randomly selects ligand for each sample
- Filters receptor atoms by ligand_ids
"""

import os
import gzip
import pickle
import numpy as np
import torch
from collections import defaultdict, OrderedDict
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path

from utils import register as R
from data.bioparse.hierarchy import merge_cplx
from data.bioparse.parser.cif_to_complex import (
    cif_to_complex,
    extract_ligand_from_cif,
    get_ligand_ids_from_receptor,
    parse_cif_file
)
from data.bioparse.utils import recur_index
from data.bioparse.interface import compute_pocket
from data.bioparse.vocab import VOCAB
from data.base import transform_data, blocks_to_data
from data.peptide import compute_terminal_rel_pos
from data.resample import ClusterResampler


@R.register('DynamicCIFDataset')
class DynamicCIFDataset:
    """
    Dynamic dataset for CIF files - loads on demand without preprocessing.
    
    Each .cif.gz file contains:
    - 1 receptor block with ligand_ids annotations
    - N ligand blocks (ligand1, ligand2, ...)
    
    On each __getitem__ call:
    1. Load .cif.gz file
    2. Randomly select a ligand
    3. Filter receptor atoms by that ligand's IDs
    4. Construct complex(receptor_filtered + ligand)
    5. Compute features including terminal positions
    """

    MODEL_KEYS = (
        'X', 'S', 'A', 'position_ids', 'chain_ids', 'is_aa',
        'block_lengths', 'bonds', 'generate_mask', 'center_mask',
        'term_rel_pos', 'lengths'
    )
    
    def __init__(
        self,
        cif_dir: str,
        cif_list: Optional[str] = None,
        cluster: Optional[str] = None,
        cluster_replace: bool = True,
        pocket_th_min: float = 10.0,
        pocket_th_max: float = 20.0,
        use_precomputed_distances: bool = False,
        precomputed_distances_dir: Optional[str] = None,
        ligand_ids_tsv: Optional[str] = None,
        max_cached_distance_files: int = 128,
        max_sample_retries: int = 10,
        seed: int = 42,
        num_workers: int = 0,
    ):
        """
        Args:
            cif_dir: Directory containing .cif.gz files
            cif_list: Path to file listing .cif.gz filenames (one per line).
                     If None, automatically discovers all .cif.gz files.
            cluster: Optional cluster file for weighted resampling between epochs.
                     Can match either the pre-filter CIF list or the final filtered list.
            cluster_replace: Whether cluster resampling should sample with replacement.
                            Default True for true weighted oversampling by cluster size.
            pocket_th_min: Minimum distance threshold in Angstroms (default: 10.0)
            pocket_th_max: Maximum distance threshold in Angstroms (default: 20.0)
                          At each __getitem__, randomly sample threshold from [pocket_th_min, pocket_th_max]
            use_precomputed_distances: If True, load pre-computed residue distances
                                      (much faster, requires preprocessing step)
                                      Files should be named: 8DTM_cluster0.distances.pkl
            precomputed_distances_dir: Directory containing precomputed distance pickle files.
                                       If None, no separate distance directory is used.
            ligand_ids_tsv: Optional TSV manifest for available ligands. The first
                            column should be the CIF basename (e.g. 8DTM_cluster0
                            or 8DTM_cluster0.cif.gz) and the second column the
                            ligand ID integer. When provided, startup skips the
                            expensive per-CIF ligand scan and trusts this metadata.
            max_cached_distance_files: Maximum number of precomputed distance
                            pickle files kept in memory per dataset instance.
                            Only used when use_precomputed_distances=True.
            max_sample_retries: Maximum number of alternate CIF/ligand samples
                            to try when one runtime-loaded sample is malformed.
            seed: Random seed for reproducibility
            num_workers: For potential parallel loading (future use)
        """
        self.cif_dir = cif_dir
        self.pocket_th_min = pocket_th_min
        self.pocket_th_max = pocket_th_max
        self.use_precomputed_distances = use_precomputed_distances
        self.precomputed_distances_dir = (
            os.path.abspath(precomputed_distances_dir)
            if precomputed_distances_dir is not None else None
        )
        self.seed = seed
        self.num_workers = num_workers
        self.cluster = cluster
        self.cluster_replace = cluster_replace
        self.ligand_ids_tsv = ligand_ids_tsv
        self.max_cached_distance_files = max(1, int(max_cached_distance_files))
        self.max_sample_retries = max(1, int(max_sample_retries))
        
        def _resolve_cif_path(entry: str) -> str:
            entry = entry.strip()
            if not entry:
                return entry
            if entry.endswith('.cif.gz') or entry.endswith('.cif'):
                return os.path.join(cif_dir, entry)
            return os.path.join(cif_dir, entry + '.cif.gz')

        def _normalize_cif_id(value: str) -> str:
            base_name = os.path.basename(value.strip())
            for suffix in ('.cif.gz', '.cif'):
                if base_name.endswith(suffix):
                    return base_name[:-len(suffix)]
            return base_name

        # Discover CIF files
        if cif_list:
            with open(cif_list, 'r') as f:
                self.cif_files = [
                    _resolve_cif_path(line)
                    for line in f if line.strip()
                ]
        else:
            # Auto-discover
            self.cif_files = sorted([
                os.path.join(cif_dir, f)
                for f in os.listdir(cif_dir)
                if f.endswith('.cif.gz') or f.endswith('.cif')
            ])

        def _load_ligand_ids_from_tsv(tsv_path: str) -> Dict[str, List[int]]:
            ligand_map = defaultdict(set)
            with open(tsv_path, 'r') as fin:
                for line_no, line in enumerate(fin, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split('\t')
                    if len(fields) < 2:
                        raise ValueError(
                            f'Expected at least 2 tab-separated columns in {tsv_path} '
                            f'at line {line_no}, got: {line}'
                        )
                    try:
                        ligand_id = int(fields[1])
                    except ValueError:
                        if line_no == 1:
                            # Allow a header row like: cif_id\tligand_id
                            continue
                        raise ValueError(
                            f'Ligand ID must be an integer in {tsv_path} at line {line_no}, '
                            f'got: {fields[1]}'
                        )
                    cif_path = _resolve_cif_path(fields[0])
                    ligand_map[cif_path].add(ligand_id)
            return {
                cif_path: sorted(ligand_ids)
                for cif_path, ligand_ids in ligand_map.items()
                if ligand_ids
            }
        
        if not self.cif_files:
            raise ValueError(f"No .cif or .cif.gz files found in {cif_dir}")
        
        print(f"Found {len(self.cif_files)} CIF files")
        print(f"Pocket threshold range: [{pocket_th_min:.1f}, {pocket_th_max:.1f}] Å")
        print(f"Using precomputed distances: {use_precomputed_distances}")
        if use_precomputed_distances:
            print(f"Precomputed distances dir: {self.precomputed_distances_dir}")
            if self.precomputed_distances_dir is None:
                print("  Warning: precomputed_distances_dir is not set; falling back to on-the-fly pocket computation.")
        
        # Pre-compute ligand IDs for each CIF file
        self.cif_ligand_ids = {}
        # Precomputed distances are loaded lazily on demand to keep memory bounded.
        self.precomputed_distances = OrderedDict()

        if ligand_ids_tsv is not None:
            print(f"Loading ligand metadata from TSV: {ligand_ids_tsv}")
            self.cif_ligand_ids = _load_ligand_ids_from_tsv(ligand_ids_tsv)
            print(f"Loaded ligand metadata for {len(self.cif_ligand_ids)} CIF files")
        else:
            print("Scanning CIF files for available ligands...")
            for cif_path in self.cif_files:
                try:
                    ligand_ids = get_ligand_ids_from_receptor(cif_path)
                    if ligand_ids:
                        self.cif_ligand_ids[cif_path] = sorted(ligand_ids)
                except Exception as e:
                    print(f"  Warning: Could not read ligands from {os.path.basename(cif_path)}: {e}")
            print(f"Successfully indexed {len(self.cif_ligand_ids)} CIF files")
        
        # Filter to only CIF files with ligands
        original_cif_files = list(self.cif_files)
        keep_indices = [i for i, f in enumerate(original_cif_files) if f in self.cif_ligand_ids]
        self.cif_files = [original_cif_files[i] for i in keep_indices]
        
        if not self.cif_files:
            raise ValueError("No CIF files with ligand information found")
        
        print(f"Using {len(self.cif_files)} CIF files with ligand data")

        if use_precomputed_distances:
            print(
                "Precomputed distances will be loaded lazily on demand "
                f"(cache size: {self.max_cached_distance_files} files)"
            )
            missing_distance_examples = []
            matched_distance_examples = []
            for cif_path in self.cif_files:
                dist_path = self._get_distance_path(cif_path)
                if dist_path is not None and os.path.exists(dist_path):
                    if len(matched_distance_examples) < 5:
                        matched_distance_examples.append(os.path.basename(dist_path))
                elif len(missing_distance_examples) < 5:
                    missing_distance_examples.append(os.path.basename(dist_path))
            if matched_distance_examples:
                print("  Example matched distance files:")
                for name in matched_distance_examples:
                    print(f"    - {name}")
            if not matched_distance_examples and missing_distance_examples:
                print("  Warning: No precomputed distance files matched. Expected names like:")
                for name in missing_distance_examples:
                    print(f"    - {name}")

        # Optional cluster-based epoch resampling, similar to the mmap datasets.
        self.resampler = None
        self.dynamic_idxs = list(range(len(self.cif_files)))
        if cluster is not None:
            resampler = ClusterResampler(cluster)
            cluster_id2prob = {
                _normalize_cif_id(_id): prob
                for _id, prob in zip(resampler.ids, resampler.idx2prob)
            }
            matched_probs = [
                cluster_id2prob[_normalize_cif_id(cif_path)]
                for cif_path in self.cif_files
                if _normalize_cif_id(cif_path) in cluster_id2prob
            ]
            if len(matched_probs) == len(self.cif_files):
                idx2prob = np.array(matched_probs, dtype=float)
            elif len(resampler.idx2prob) == len(original_cif_files):
                idx2prob = resampler.idx2prob[keep_indices]
            elif len(resampler.idx2prob) == len(self.cif_files):
                idx2prob = resampler.idx2prob
            else:
                missing_cluster = [
                    _normalize_cif_id(cif_path)
                    for cif_path in self.cif_files
                    if _normalize_cif_id(cif_path) not in cluster_id2prob
                ][:5]
                raise ValueError(
                    f"Could not align cluster file {os.path.basename(cluster)} to the CIF list. "
                    f"Matched {len(matched_probs)}/{len(self.cif_files)} CIF IDs by name. "
                    f"Examples missing from cluster file: {missing_cluster}"
                )
            idx2prob = idx2prob / idx2prob.sum()
            resampler.idx2prob = idx2prob
            self.resampler = resampler
            print(
                f"Using cluster resampling from {os.path.basename(cluster)} "
                f"(replace={cluster_replace})"
            )
            self.update_epoch()

    @staticmethod
    def _get_distance_filename(cif_path: str) -> str:
        base_name = os.path.basename(cif_path)
        if base_name.endswith('.cif.gz'):
            return base_name[:-7] + '.distances.pkl'
        if base_name.endswith('.cif'):
            return base_name[:-4] + '.distances.pkl'
        return base_name + '.distances.pkl'

    def _get_distance_path(self, cif_path: str) -> Optional[str]:
        if self.precomputed_distances_dir is None:
            return None
        base_name = os.path.basename(cif_path)
        candidates = [self._get_distance_filename(cif_path)]
        candidates.append(base_name + '.distances.pkl')
        normalized_name = base_name
        for suffix in ('.cif.gz', '.cif'):
            if normalized_name.endswith(suffix):
                normalized_name = normalized_name[:-len(suffix)]
                break
        normalized_name = normalized_name + '.distances.pkl'
        if normalized_name not in candidates:
            candidates.append(normalized_name)
        for candidate in candidates:
            dist_path = os.path.join(self.precomputed_distances_dir, candidate)
            if os.path.exists(dist_path):
                return dist_path
        return os.path.join(self.precomputed_distances_dir, candidates[0])
    
    def __len__(self) -> int:
        """Total number of CIF files. Each file can be sampled multiple times with different ligands."""
        return len(self.dynamic_idxs)

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(
                len(self.cif_files),
                replace=self.cluster_replace
            ).tolist()
        else:
            self.dynamic_idxs = list(range(len(self.cif_files)))

    def _get_precomputed_distances_for_cif(self, cif_path: str):
        if not self.use_precomputed_distances:
            return None
        if cif_path in self.precomputed_distances:
            self.precomputed_distances.move_to_end(cif_path)
            return self.precomputed_distances[cif_path]
        dist_path = self._get_distance_path(cif_path)
        if dist_path is None or not os.path.exists(dist_path):
            return None
        try:
            with open(dist_path, 'rb') as f:
                distances_data = pickle.load(f)
        except Exception as e:
            print(f"  Warning: Could not load distances from {os.path.basename(dist_path)}: {e}")
            return None
        self.precomputed_distances[cif_path] = distances_data
        self.precomputed_distances.move_to_end(cif_path)
        while len(self.precomputed_distances) > self.max_cached_distance_files:
            self.precomputed_distances.popitem(last=False)
        return distances_data
    
    def _load_cif_complex(self, cif_idx: int, ligand_id: int) -> tuple:
        """
        Load receptor and ligand from CIF file, filtering receptor by ligand_ids.
        
        Returns:
            (merged_cplx, receptor_cplx, ligand_cplx, receptor_blocks, ligand_blocks)
        """
        cif_path = self.cif_files[cif_idx]
        
        # Parse receptor with ligand_ids filter
        receptor_cplx = cif_to_complex(
            cif_path,
            ligand_ids_filter={ligand_id}
        )
        
        # Extract specific ligand structure
        ligand_cplx = extract_ligand_from_cif(cif_path, ligand_id)
        
        # Get blocks
        receptor_chain = list(receptor_cplx.id2idx.keys())[0] if receptor_cplx else None
        ligand_chain = list(ligand_cplx.id2idx.keys())[0] if ligand_cplx else None
        
        receptor_blocks = receptor_cplx[receptor_chain].blocks if receptor_chain else []
        ligand_blocks = ligand_cplx[ligand_chain].blocks if ligand_chain else []
        merged_cplx = merge_cplx(receptor_cplx, ligand_cplx)
        
        return merged_cplx, receptor_cplx, ligand_cplx, receptor_blocks, ligand_blocks
    
    def _get_pocket_blocks_from_precomputed(
        self,
        cif_path: str,
        ligand_id: int,
        pocket_th: float,
        receptor_cplx
    ) -> List[Tuple]:
        """
        Get pocket block indices using precomputed distances.
        
        Args:
            cif_path: Path to CIF file
            ligand_id: Which ligand to compute pocket for
            pocket_th: Distance threshold in Angstroms
            receptor_cplx: Receptor complex object
            
        Returns:
            List of block indices (chain, block_id) that are within pocket_th
        """
        distances_data = self._get_precomputed_distances_for_cif(cif_path)
        if distances_data is None:
            return None

        if ligand_id not in distances_data:
            return None
        
        ligand_distances = distances_data[ligand_id]
        
        # Get all receptor residues and filter by distance
        receptor_chain = list(receptor_cplx.id2idx.keys())[0]
        receptor_blocks = receptor_cplx[receptor_chain].blocks
        
        pocket_indices = []
        for block_idx, block in enumerate(receptor_blocks):
            if block.id in ligand_distances and ligand_distances[block.id] <= pocket_th:
                pocket_indices.append((receptor_chain, block.id))
            elif block_idx in ligand_distances and ligand_distances[block_idx] <= pocket_th:
                pocket_indices.append((receptor_chain, block.id))
        
        return pocket_indices

    @staticmethod
    def _empty_sample(error: str = "") -> Dict:
        return {
            'X': torch.empty((0, 3), dtype=torch.float),
            'S': torch.empty(0, dtype=torch.long),
            'A': torch.empty(0, dtype=torch.long),
            'position_ids': torch.empty(0, dtype=torch.long),
            'chain_ids': torch.empty(0, dtype=torch.long),
            'is_aa': torch.empty(0, dtype=torch.bool),
            'block_lengths': torch.empty(0, dtype=torch.long),
            'bonds': torch.empty((0, 3), dtype=torch.long),
            'generate_mask': torch.empty(0, dtype=torch.bool),
            'center_mask': torch.empty(0, dtype=torch.bool),
            'term_rel_pos': torch.empty((0, 15), dtype=torch.float),
            'lengths': 0,
            '_valid': False,
            '_error': error,
        }

    @staticmethod
    def _sample_validation_error(data: Dict) -> Optional[str]:
        if not isinstance(data, dict):
            return f"sample is not a dict: {type(data)}"
        missing_keys = [key for key in DynamicCIFDataset.MODEL_KEYS if key not in data]
        if missing_keys:
            return f"missing required keys: {missing_keys}"

        try:
            n_blocks = int(data['lengths'])
            n_atoms = int(data['A'].shape[0])
            if n_blocks <= 0 or n_atoms <= 0:
                return f"empty sample: lengths={n_blocks}, atoms={n_atoms}"
            if not torch.is_floating_point(data['X']):
                return f"X dtype is {data['X'].dtype}, expected floating point"
            if data['X'].ndim != 2 or data['X'].shape != (n_atoms, 3):
                return f"X shape {tuple(data['X'].shape)} does not match atoms={n_atoms}"
            if not bool(torch.isfinite(data['X']).all().item()):
                return "X contains NaN or Inf"
            if data['S'].shape[0] != n_blocks:
                return f"S shape {tuple(data['S'].shape)} does not match blocks={n_blocks}"
            if data['S'].numel() != 0 and int(data['S'].min().item()) < 0:
                return "S contains negative block type IDs"
            if data['A'].numel() != 0 and int(data['A'].min().item()) < 0:
                return "A contains negative atom type IDs"
            if data['position_ids'].shape[0] != n_blocks:
                return f"position_ids shape {tuple(data['position_ids'].shape)} does not match blocks={n_blocks}"
            if data['chain_ids'].shape[0] != n_blocks:
                return f"chain_ids shape {tuple(data['chain_ids'].shape)} does not match blocks={n_blocks}"
            if data['is_aa'].shape[0] != n_blocks:
                return f"is_aa shape {tuple(data['is_aa'].shape)} does not match blocks={n_blocks}"
            if data['block_lengths'].shape[0] != n_blocks:
                return f"block_lengths shape {tuple(data['block_lengths'].shape)} does not match blocks={n_blocks}"
            if int(data['block_lengths'].sum().item()) != n_atoms:
                return (
                    f"sum(block_lengths)={int(data['block_lengths'].sum().item())} "
                    f"does not match atoms={n_atoms}"
                )
            if data['generate_mask'].shape[0] != n_blocks:
                return f"generate_mask shape {tuple(data['generate_mask'].shape)} does not match blocks={n_blocks}"
            if data['generate_mask'].dtype != torch.bool:
                return f"generate_mask dtype is {data['generate_mask'].dtype}, expected bool"
            if data['center_mask'].shape[0] != n_blocks:
                return f"center_mask shape {tuple(data['center_mask'].shape)} does not match blocks={n_blocks}"
            if data['center_mask'].dtype != torch.bool:
                return f"center_mask dtype is {data['center_mask'].dtype}, expected bool"
            if not bool(data['generate_mask'].any().item()):
                return "generate_mask has no generated blocks"
            if not bool(data['center_mask'].any().item()):
                return "center_mask has no context/centering blocks"
            if data['term_rel_pos'].shape != (n_blocks, 15):
                return f"term_rel_pos shape {tuple(data['term_rel_pos'].shape)} does not match ({n_blocks}, 15)"
            if not torch.is_floating_point(data['term_rel_pos']):
                return f"term_rel_pos dtype is {data['term_rel_pos'].dtype}, expected floating point"
            if not bool(torch.isfinite(data['term_rel_pos']).all().item()):
                return "term_rel_pos contains NaN or Inf"
            if data['bonds'].ndim != 2 or data['bonds'].shape[1] != 3:
                return f"bonds shape {tuple(data['bonds'].shape)} is not [Nbonds, 3]"
            if data['bonds'].dtype != torch.long:
                return f"bonds dtype is {data['bonds'].dtype}, expected long"
            if data['bonds'].numel() != 0:
                if int(data['bonds'][:, :2].min().item()) < 0:
                    return "bonds contain negative atom indices"
                if int(data['bonds'][:, :2].max().item()) >= n_atoms:
                    return (
                        f"bonds reference atom index {int(data['bonds'][:, :2].max().item())} "
                        f"but sample has only {n_atoms} atoms"
                    )
        except Exception:
            return "sample validation raised an unexpected exception"

        return None

    @staticmethod
    def _is_valid_sample(data: Dict) -> bool:
        return DynamicCIFDataset._sample_validation_error(data) is None

    @staticmethod
    def _sample_debug_id(data: Dict) -> str:
        if not isinstance(data, dict):
            return f"type={type(data)}"
        cif_path = data.get('_cif_path', '?')
        cif_name = os.path.basename(str(cif_path)) if cif_path != '?' else '?'
        return (
            f"cif={cif_name}, ligand={data.get('_ligand_id', '?')}, "
            f"dataset_idx={data.get('_dataset_idx', '?')}, cif_idx={data.get('_cif_idx', '?')}"
        )
    
    def _build_sample(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the CIF file
            
        Returns:
            Data dictionary with features for training/inference
        """
        requested_idx = idx
        cif_idx = self.dynamic_idxs[idx]
        cif_path = self.cif_files[cif_idx]
        available_ligands = self.cif_ligand_ids[cif_path]
        
        # Sample ligand/threshold from the worker-local torch RNG so training
        # remains reproducible without freezing one ligand choice per file.
        ligand_idx = torch.randint(len(available_ligands), (1,)).item()
        ligand_id = available_ligands[ligand_idx]
        
        # Randomly sample pocket threshold from [pocket_th_min, pocket_th_max]
        pocket_th = self.pocket_th_min + (
            (self.pocket_th_max - self.pocket_th_min) * torch.rand(1).item()
        )
        
        try:
            merged_cplx, receptor_cplx, ligand_cplx, receptor_blocks, ligand_blocks = \
                self._load_cif_complex(cif_idx, ligand_id)
        except Exception as e:
            raise RuntimeError(
                f"Error loading CIF {os.path.basename(cif_path)}, ligand {ligand_id}: {e}"
            ) from e

        if not receptor_blocks:
            raise RuntimeError(
                f"No receptor blocks parsed for CIF {os.path.basename(cif_path)}, ligand {ligand_id}"
            )
        if not ligand_blocks:
            raise RuntimeError(
                f"No ligand blocks parsed for CIF {os.path.basename(cif_path)}, ligand {ligand_id}"
            )
        
        # Compute pocket using precomputed distances if available
        receptor_chain = list(receptor_cplx.id2idx.keys())[0]
        ligand_chain = list(ligand_cplx.id2idx.keys())[0]
        
        # Try precomputed distances first
        pocket_block_ids = None
        if self.use_precomputed_distances:
            pocket_block_ids = self._get_pocket_blocks_from_precomputed(
                cif_path, ligand_id, pocket_th, receptor_cplx
            )
        
        # Fall back to on-the-fly computation if no precomputed distances
        if pocket_block_ids is None:
            pocket_block_ids, _ = compute_pocket(
                merged_cplx,
                [receptor_chain],
                [ligand_chain],
                dist_th=pocket_th
            )
        
        # CFG training requires at least some pocket context
        # If no pocket blocks found, resample with larger threshold
        if not pocket_block_ids:
            pocket_th = max(self.pocket_th_max, pocket_th + 5.0)  # Increase threshold
            pocket_block_ids, _ = compute_pocket(
                merged_cplx,
                [receptor_chain],
                [ligand_chain],
                dist_th=pocket_th
            )
        
        # Combine receptor and ligand blocks
        all_select_indexes = pocket_block_ids + [
            (ligand_chain, block.id) for block in ligand_blocks
        ]
        
        # Transform to tensors
        try:
            data = transform_data(merged_cplx, all_select_indexes)
        except Exception as e:
            raise RuntimeError(
                f"Error transforming CIF {os.path.basename(cif_path)}, ligand {ligand_id}: {e}"
            ) from e
        
        # Add masks
        n_pocket_blocks = len(pocket_block_ids)
        n_ligand_blocks = len(ligand_blocks)
        if n_pocket_blocks == 0:
            raise RuntimeError(
                f"No pocket blocks found for CIF {os.path.basename(cif_path)}, ligand {ligand_id}"
            )
        if n_ligand_blocks == 0:
            raise RuntimeError(
                f"No ligand blocks found for CIF {os.path.basename(cif_path)}, ligand {ligand_id}"
            )
        expected_blocks = n_pocket_blocks + n_ligand_blocks
        if int(data['lengths']) != expected_blocks:
            raise RuntimeError(
                f"Selected block count mismatch for CIF {os.path.basename(cif_path)}, "
                f"ligand {ligand_id}: selected {expected_blocks} blocks "
                f"({n_pocket_blocks} pocket + {n_ligand_blocks} ligand), "
                f"but transform_data kept {int(data['lengths'])}. This usually means "
                "one selected block had no modeled heavy atoms."
            )
        generate_mask = [0] * n_pocket_blocks + [1] * n_ligand_blocks
        center_mask = [1] * n_pocket_blocks + [0] * n_ligand_blocks
        
        data['generate_mask'] = torch.tensor(generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(center_mask, dtype=torch.bool)
        
        # Compute one endpoint-amide condition and copy it to every ligand block.
        ligand_term_rel_pos = compute_terminal_rel_pos(ligand_blocks)
        term_pos_list = [
            *[np.zeros(15, dtype=np.float32) for _ in range(n_pocket_blocks)],
            *[ligand_term_rel_pos for _ in range(n_ligand_blocks)],
        ]

        data['term_rel_pos'] = torch.tensor(
            np.array(term_pos_list, dtype=np.float32),
            dtype=torch.float
        )

        # Keep generated ligand positions contiguous, matching the peptide dataset behavior.
        gen_mask = data['generate_mask']
        ligand_position_ids = data['position_ids'][gen_mask]
        ligand_position_ids = ligand_position_ids - ligand_position_ids.min() + 1
        data['position_ids'][gen_mask] = ligand_position_ids
        
        # Add metadata for CFG training and logging
        data['pocket_th'] = pocket_th
        data['cfg_eligible'] = n_pocket_blocks > 0  # Can this sample use CFG dropout?
        data['_valid'] = True
        data['_dataset_idx'] = requested_idx
        data['_cif_idx'] = cif_idx
        data['_cif_path'] = cif_path
        data['_ligand_id'] = ligand_id

        validation_error = self._sample_validation_error(data)
        if validation_error is not None:
            raise RuntimeError(
                f"Invalid sample for CIF {os.path.basename(cif_path)}, "
                f"ligand {ligand_id}: {validation_error}"
            )
        
        return data

    def __getitem__(self, idx: int) -> Dict:
        last_error = ""
        for attempt in range(self.max_sample_retries):
            cur_idx = idx
            if attempt > 0:
                cur_idx = torch.randint(len(self), (1,)).item()
            try:
                data = self._build_sample(cur_idx)
                if self._is_valid_sample(data):
                    return data
                last_error = self._sample_validation_error(data) or "sample failed schema/shape validation"
            except Exception as e:
                last_error = str(e)

            if attempt == 0:
                print(
                    f"  Warning: Bad DynamicCIFDataset sample at dataset index {idx}: "
                    f"{last_error}. Retrying with alternate samples."
                )

        return self._empty_sample(
            f"Failed to build a valid sample after {self.max_sample_retries} attempts; "
            f"last error: {last_error}"
        )
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.
        
        Args:
            batch: List of data dictionaries from __getitem__
            
        Returns:
            Batched data dictionary
        """
        target_batch_size = len(batch)
        valid_batch = []
        invalid_errors = []
        for item in batch:
            if self._is_valid_sample(item):
                valid_batch.append(item)
            else:
                error = self._sample_validation_error(item)
                if isinstance(item, dict) and item.get('_error'):
                    error = item['_error']
                invalid_errors.append(f"{self._sample_debug_id(item)}: {error}")

        # A bad CIF/ligand should not crash a long run. Replace malformed items
        # with alternate valid samples from the same dataset whenever possible.
        replacement_budget = target_batch_size * self.max_sample_retries
        while len(valid_batch) < target_batch_size and replacement_budget > 0:
            replacement_budget -= 1
            replacement_idx = torch.randint(len(self), (1,)).item()
            replacement = self[replacement_idx]
            if self._is_valid_sample(replacement):
                valid_batch.append(replacement)

        duplicated_replacements = 0
        while len(valid_batch) < target_batch_size and valid_batch:
            duplicate_idx = torch.randint(len(valid_batch), (1,)).item()
            valid_batch.append(valid_batch[duplicate_idx])
            duplicated_replacements += 1

        if invalid_errors:
            print(
                f"  Warning: Replaced {len(invalid_errors)} invalid DynamicCIFDataset "
                f"samples during collation. First error: {invalid_errors[0]}"
            )

        if not valid_batch:
            raise RuntimeError(
                "All DynamicCIFDataset samples in this batch were invalid, and no "
                "valid replacement could be drawn. Please inspect the warned CIF/ligand cases."
            )
        if duplicated_replacements:
            print(
                f"  Warning: Duplicated {duplicated_replacements} valid "
                "DynamicCIFDataset samples to preserve batch size after replacement attempts."
            )

        batch = valid_batch
        results = {}
        for key in self.MODEL_KEYS:
            values = [item[key] for item in batch]
            
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'bonds':
                # Add atom offset to bond indices
                offset = 0
                for i, bonds in enumerate(values):
                    bonds = bonds.clone()
                    if bonds.numel() == 0:
                        bonds = bonds.reshape(0, 3)
                    else:
                        bonds[:, :2] = bonds[:, :2] + offset
                    if i == 0:
                        results[key] = bonds
                    else:
                        results[key] = torch.cat([results[key], bonds], dim=0)
                    offset += len(batch[i]['A'])
            else:
                results[key] = torch.cat(values, dim=0)
        
        return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dynamic_cif_dataset.py <cif_dir>")
        sys.exit(1)
    
    cif_dir = sys.argv[1]
    
    # Test dataset
    dataset = DynamicCIFDataset(cif_dir, pocket_th_min=10.0, pocket_th_max=10.0)
    print(f"Dataset size: {len(dataset)}")
    
    # Try loading first sample
    try:
        sample = dataset[0]
        print(f"Sample 0:")
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape} {val.dtype}")
            else:
                print(f"  {key}: {val}")
    except Exception as e:
        print(f"Error loading sample: {e}")
        import traceback
        traceback.print_exc()
