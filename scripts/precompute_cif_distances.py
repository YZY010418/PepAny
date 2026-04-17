#!/usr/bin/env python3
"""
Pre-compute residue-ligand distances for fast pocket detection at runtime.

This script pre-computes the minimum distance from each receptor residue
to each ligand structure. At runtime, we can then quickly filter residues
by thresholds without recalculating distances.

Usage:
    python3 scripts/precompute_cif_distances.py <cif_dir> [--output-dir <output_dir>]

Output files:
    <cif_file>.distances.pkl - pickle file with distances
        Format: {
            ligand_id: {residue_block_id: min_distance, ...},
            ...
        }
"""

import os
import sys
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
from pathlib import Path
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


def compute_distances_for_cif(cif_path, output_dir):
    """
    Compute distances for a single CIF file.
    
    Args:
        cif_path: Path to .cif.gz file
        output_dir: Where to save .distances.pkl
        
    Returns:
        Path to output file, or None if failed
    """
    try:
        # Import here to avoid issues if module not fully loaded
        repo_root = str(REPO_ROOT)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from data.bioparse.parser.cif_to_complex import (
            cif_to_complex,
            extract_ligand_from_cif,
            get_ligand_ids_from_receptor
        )
        from data.bioparse.interface import blocks_to_cb_coords
        
        # Parse receptor
        receptor = cif_to_complex(cif_path)
        
        # Get available ligands
        ligand_ids = get_ligand_ids_from_receptor(cif_path)
        if not ligand_ids:
            return None
        
        receptor_chain = list(receptor.id2idx.keys())[0]
        receptor_blocks = receptor[receptor_chain].blocks
        
        # Use the same representative block coordinates as runtime pocket detection:
        # Cb/virtual-Cb for amino acids and center-of-mass for non-proteins.
        receptor_coords = blocks_to_cb_coords(receptor_blocks).astype(np.float32)
        
        # Compute distances to each ligand
        distances_data = {}
        
        for ligand_id in sorted(ligand_ids):
            try:
                ligand = extract_ligand_from_cif(cif_path, ligand_id)
                ligand_chain = list(ligand.id2idx.keys())[0]
                ligand_blocks = ligand[ligand_chain].blocks
                
                if not ligand_blocks:
                    continue
                
                ligand_coords = blocks_to_cb_coords(ligand_blocks).astype(np.float32)
                
                # Compute the same block-level minimum distances used by compute_pocket().
                residue_distances = {}
                for block, res_coord in zip(receptor_blocks, receptor_coords):
                    distances = np.linalg.norm(ligand_coords - res_coord, axis=1)
                    residue_distances[block.id] = float(np.min(distances))
                
                distances_data[ligand_id] = residue_distances
                
            except Exception as e:
                print(f"  Warning: Could not process ligand {ligand_id}: {e}")
        
        if not distances_data:
            return None
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.basename(cif_path)
        if base_name.endswith('.cif.gz'):
            out_name = base_name[:-7] + '.distances.pkl'
        elif base_name.endswith('.cif'):
            out_name = base_name[:-4] + '.distances.pkl'
        else:
            out_name = base_name + '.distances.pkl'
        
        out_path = os.path.join(output_dir, out_name)
        
        with open(out_path, 'wb') as f:
            pickle.dump(distances_data, f)
        
        return out_path
        
    except Exception as e:
        print(f"Error processing {os.path.basename(cif_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute residue-ligand distances for fast pocket detection'
    )
    parser.add_argument('cif_dir', help='Directory containing .cif.gz files')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for .distances.pkl files (default: same as cif_dir)')
    parser.add_argument('--pattern', default='*.cif.gz',
                       help='File pattern to match (default: *.cif.gz)')
    parser.add_argument(
        '-j', '--num-workers',
        type=int,
        default=max(1, os.cpu_count() or 1),
        help='Number of CPU worker processes to use (default: all available CPUs)'
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=1,
        help='Task chunksize for multiprocessing (default: 1)'
    )
    
    args = parser.parse_args()
    
    cif_dir = args.cif_dir
    output_dir = args.output_dir or cif_dir
    
    if not os.path.isdir(cif_dir):
        print(f"Error: Directory not found: {cif_dir}")
        sys.exit(1)
    
    # Find CIF files
    cif_files = sorted(Path(cif_dir).glob(args.pattern))
    
    if not cif_files:
        print(f"No files matching '{args.pattern}' found in {cif_dir}")
        sys.exit(1)
    
    print(f"Found {len(cif_files)} CIF files to process")
    print(f"Output directory: {output_dir}")
    print(f"CPU workers: {args.num_workers}")
    print()
    
    # Process each file
    successful = 0
    failed = 0

    cif_paths = [str(path) for path in cif_files]
    num_workers = max(1, min(args.num_workers, len(cif_paths)))

    if num_workers == 1:
        result_iter = (
            compute_distances_for_cif(cif_path, output_dir)
            for cif_path in cif_paths
        )
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        result_iter = executor.map(
            compute_distances_for_cif,
            cif_paths,
            repeat(output_dir),
            chunksize=max(1, args.chunksize)
        )

    try:
        for out_path in tqdm(result_iter, total=len(cif_paths), desc="Computing distances"):
            if out_path:
                successful += 1
            else:
                failed += 1
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True)
    
    print()
    print(f"✓ Completed: {successful} successful, {failed} failed")
    print(f"Output files saved to: {output_dir}")
    print()
    print("To use precomputed distances in training:")
    print("  dataset = DynamicCIFDataset(")
    print(f"    cif_dir='{cif_dir}',")
    print("    use_precomputed_distances=True,")
    print("  )")


if __name__ == '__main__':
    main()
