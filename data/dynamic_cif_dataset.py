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
import random
import pickle
import numpy as np
import torch
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path

from utils import register as R
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
from data.peptide import (
    terminal_descriptor,
    compute_term_rel_pos
)


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
    
    def __init__(
        self,
        cif_dir: str,
        cif_list: Optional[str] = None,
        pocket_th_min: float = 10.0,
        pocket_th_max: float = 20.0,
        use_precomputed_distances: bool = False,
        seed: int = 42,
        num_workers: int = 0,
    ):
        """
        Args:
            cif_dir: Directory containing .cif.gz files
            cif_list: Path to file listing .cif.gz filenames (one per line).
                     If None, automatically discovers all .cif.gz files.
            pocket_th_min: Minimum distance threshold in Angstroms (default: 10.0)
            pocket_th_max: Maximum distance threshold in Angstroms (default: 20.0)
                          At each __getitem__, randomly sample threshold from [pocket_th_min, pocket_th_max]
            use_precomputed_distances: If True, load pre-computed residue distances
                                      (much faster, requires preprocessing step)
                                      Files should be named: 8DTM_cluster0.distances.pkl
            seed: Random seed for reproducibility
            num_workers: For potential parallel loading (future use)
        """
        self.cif_dir = cif_dir
        self.pocket_th_min = pocket_th_min
        self.pocket_th_max = pocket_th_max
        self.use_precomputed_distances = use_precomputed_distances
        self.seed = seed
        self.num_workers = num_workers
        
        # Discover CIF files
        if cif_list:
            with open(cif_list, 'r') as f:
                self.cif_files = [
                    os.path.join(cif_dir, line.strip())
                    for line in f if line.strip()
                ]
        else:
            # Auto-discover
            self.cif_files = sorted([
                os.path.join(cif_dir, f)
                for f in os.listdir(cif_dir)
                if f.endswith('.cif.gz') or f.endswith('.cif')
            ])
        
        if not self.cif_files:
            raise ValueError(f"No .cif or .cif.gz files found in {cif_dir}")
        
        print(f"Found {len(self.cif_files)} CIF files")
        print(f"Pocket threshold range: [{pocket_th_min:.1f}, {pocket_th_max:.1f}] Å")
        print(f"Using precomputed distances: {use_precomputed_distances}")
        
        # Pre-compute ligand IDs for each CIF file
        self.cif_ligand_ids = {}
        # Load precomputed distances if available
        self.precomputed_distances = {}
        
        print("Scanning CIF files for available ligands...")
        for cif_path in self.cif_files:
            try:
                ligand_ids = get_ligand_ids_from_receptor(cif_path)
                if ligand_ids:
                    self.cif_ligand_ids[cif_path] = sorted(ligand_ids)
                    
                    # Try to load precomputed distances
                    if use_precomputed_distances:
                        dist_path = cif_path.replace('.cif.gz', '.distances.pkl').replace('.cif', '.distances.pkl')
                        if os.path.exists(dist_path):
                            try:
                                with open(dist_path, 'rb') as f:
                                    self.precomputed_distances[cif_path] = pickle.load(f)
                            except Exception as e:
                                print(f"  Warning: Could not load distances from {os.path.basename(dist_path)}: {e}")
                        
            except Exception as e:
                print(f"  Warning: Could not read ligands from {os.path.basename(cif_path)}: {e}")
        
        print(f"Successfully indexed {len(self.cif_ligand_ids)} CIF files")
        print(f"Precomputed distances available for {len(self.precomputed_distances)} files")
        
        # Filter to only CIF files with ligands
        self.cif_files = [f for f in self.cif_files if f in self.cif_ligand_ids]
        
        if not self.cif_files:
            raise ValueError("No CIF files with ligand information found")
        
        print(f"Using {len(self.cif_files)} CIF files with ligand data")
    
    def __len__(self) -> int:
        """Total number of CIF files. Each file can be sampled multiple times with different ligands."""
        return len(self.cif_files)
    
    def _load_cif_complex(self, cif_idx: int, ligand_id: int) -> tuple:
        """
        Load receptor and ligand from CIF file, filtering receptor by ligand_ids.
        
        Returns:
            (receptor_cplx, ligand_cplx, receptor_blocks, ligand_blocks)
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
        
        return receptor_cplx, ligand_cplx, receptor_blocks, ligand_blocks
    
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
        if cif_path not in self.precomputed_distances:
            return None
        
        distances_data = self.precomputed_distances[cif_path]
        
        if ligand_id not in distances_data:
            return None
        
        ligand_distances = distances_data[ligand_id]  # dict: residue_idx → min_distance
        
        # Get all receptor residues and filter by distance
        receptor_chain = list(receptor_cplx.id2idx.keys())[0]
        receptor_blocks = receptor_cplx[receptor_chain].blocks
        
        pocket_indices = []
        for block_idx, block in enumerate(receptor_blocks):
            if block_idx in ligand_distances and ligand_distances[block_idx] <= pocket_th:
                pocket_indices.append((receptor_chain, block.id))
        
        return pocket_indices
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the CIF file
            
        Returns:
            Data dictionary with features for training/inference
        """
        # Set seed for reproducibility within this sample
        random.seed(self.seed + idx)
        np.random.seed(self.seed + idx)
        torch.manual_seed(self.seed + idx)
        
        cif_path = self.cif_files[idx]
        available_ligands = self.cif_ligand_ids[cif_path]
        
        # Randomly select a ligand
        ligand_id = random.choice(available_ligands)
        
        # Randomly sample pocket threshold from [pocket_th_min, pocket_th_max]
        pocket_th = random.uniform(self.pocket_th_min, self.pocket_th_max)
        
        try:
            receptor_cplx, ligand_cplx, receptor_blocks, ligand_blocks = \
                self._load_cif_complex(idx, ligand_id)
        except Exception as e:
            print(f"Error loading CIF {os.path.basename(cif_path)}, ligand {ligand_id}: {e}")
            # Return empty sample on error
            return {
                'X': torch.empty((0, 3), dtype=torch.float),
                'S': torch.empty(0, dtype=torch.long),
                'A': torch.empty(0, dtype=torch.long),
                'position_ids': torch.empty(0, dtype=torch.long),
                'chain_ids': torch.empty(0, dtype=torch.long),
                'is_aa': torch.empty(0, dtype=torch.bool),
                'block_lengths': torch.empty(0, dtype=torch.long),
                'bonds': torch.empty((0, 3), dtype=torch.long),
                'term_rel_pos': torch.empty((0, 15), dtype=torch.float),
                'lengths': 0,
                'pocket_th': pocket_th,
            }
        
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
                receptor_cplx,
                [receptor_chain],
                [ligand_chain],
                dist_th=pocket_th
            )
        
        # CFG training requires at least some pocket context
        # If no pocket blocks found, resample with larger threshold
        if not pocket_block_ids:
            pocket_th = max(self.pocket_th_max, pocket_th + 5.0)  # Increase threshold
            pocket_block_ids, _ = compute_pocket(
                receptor_cplx,
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
            data = transform_data(receptor_cplx, all_select_indexes)
        except Exception as e:
            print(f"Error transforming data: {e}")
            return {
                'X': torch.empty((0, 3), dtype=torch.float),
                'S': torch.empty(0, dtype=torch.long),
                'A': torch.empty(0, dtype=torch.long),
                'position_ids': torch.empty(0, dtype=torch.long),
                'chain_ids': torch.empty(0, dtype=torch.long),
                'is_aa': torch.empty(0, dtype=torch.bool),
                'block_lengths': torch.empty(0, dtype=torch.long),
                'bonds': torch.empty((0, 3), dtype=torch.long),
                'term_rel_pos': torch.empty((0, 15), dtype=torch.float),
                'lengths': 0,
                'pocket_th': pocket_th,
            }
        
        # Add masks
        n_pocket_blocks = len(pocket_block_ids)
        n_ligand_blocks = len(ligand_blocks)
        generate_mask = [0] * n_pocket_blocks + [1] * n_ligand_blocks
        center_mask = [1] * n_pocket_blocks + [0] * n_ligand_blocks
        
        data['generate_mask'] = torch.tensor(generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(center_mask, dtype=torch.bool)
        
        # Compute terminal positions for ligand blocks
        term_pos_list = []
        
        # Pocket: zeros
        for _ in range(n_pocket_blocks):
            term_pos_list.append(np.zeros(15, dtype=np.float32))
        
        # Ligand: compute terminal positions
        for i, block in enumerate(ligand_blocks):
            block_n = ligand_blocks[i-1] if i > 0 else None
            block_c = ligand_blocks[i+1] if i < len(ligand_blocks)-1 else None
            term_pos_list.append(compute_term_rel_pos(block_n, block, block_c))
        
        data['term_rel_pos'] = torch.tensor(
            np.array(term_pos_list, dtype=np.float32),
            dtype=torch.float
        )
        
        # Add metadata for CFG training and logging
        data['pocket_th'] = pocket_th
        data['cfg_eligible'] = n_pocket_blocks > 0  # Can this sample use CFG dropout?
        
        return data
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.
        
        Args:
            batch: List of data dictionaries from __getitem__
            
        Returns:
            Batched data dictionary
        """
        results = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            
            # Skip metadata fields (keep as lists for logging)
            if key in ['pocket_th', 'cfg_eligible']:
                results[key] = values
            elif key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'bonds':
                # Add atom offset to bond indices
                offset = 0
                for i, bonds in enumerate(values):
                    bonds = bonds.clone()
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
    dataset = DynamicCIFDataset(cif_dir, pocket_th=10.0)
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
