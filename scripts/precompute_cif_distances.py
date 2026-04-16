#!/usr/bin/env python3
"""
Pre-compute residue-ligand distances for fast pocket detection at runtime.

This script pre-computes the minimum distance from each receptor residue
to each ligand structure. At runtime, we can then quickly filter residues
by thresholds without recalculating distances.

Usage:
    python3 precompute_distances.py <cif_dir> [--output-dir <output_dir>]

Output files:
    <cif_file>.distances.pkl - pickle file with distances
        Format: {
            ligand_id: {residue_idx: min_distance, ...},
            ...
        }
"""

import os
import sys
import pickle
import gzip
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_residue_atoms(block):
    """Get CA coordinates from a block (residue)"""
    for atom in block:
        if atom.name == 'CA':
            return np.array(atom.get_coord(), dtype=np.float32)
    # Fallback: average of all atoms
    coords = []
    for atom in block:
        coords.append(np.array(atom.get_coord(), dtype=np.float32))
    if coords:
        return np.mean(coords, axis=0)
    return None


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
        sys.path.insert(0, '/Users/bytedance/Desktop/PepAny')
        from data.bioparse.parser.cif_to_complex import (
            cif_to_complex,
            extract_ligand_from_cif,
            get_ligand_ids_from_receptor
        )
        
        # Parse receptor
        receptor = cif_to_complex(cif_path)
        
        # Get available ligands
        ligand_ids = get_ligand_ids_from_receptor(cif_path)
        if not ligand_ids:
            return None
        
        receptor_chain = list(receptor.id2idx.keys())[0]
        receptor_blocks = receptor[receptor_chain].blocks
        
        # Get receptor residue coordinates (CA atoms)
        receptor_coords = []
        for block in receptor_blocks:
            ca_coord = get_residue_atoms(block)
            if ca_coord is not None:
                receptor_coords.append(ca_coord)
            else:
                receptor_coords.append(np.zeros(3))
        
        # Compute distances to each ligand
        distances_data = {}
        
        for ligand_id in sorted(ligand_ids):
            try:
                ligand = extract_ligand_from_cif(cif_path, ligand_id)
                ligand_chain = list(ligand.id2idx.keys())[0]
                ligand_blocks = ligand[ligand_chain].blocks
                
                # Get ligand atom coordinates (all atoms)
                ligand_coords = []
                for block in ligand_blocks:
                    for atom in block:
                        ligand_coords.append(np.array(atom.get_coord(), dtype=np.float32))
                
                if not ligand_coords:
                    continue
                
                ligand_coords = np.array(ligand_coords)
                
                # Compute minimum distance from each residue to ligand atoms
                residue_distances = {}
                for res_idx, res_coord in enumerate(receptor_coords):
                    if np.allclose(res_coord, 0):  # Empty residue
                        residue_distances[res_idx] = float('inf')
                    else:
                        # Minimum distance from this residue to any ligand atom
                        distances = np.linalg.norm(ligand_coords - res_coord, axis=1)
                        residue_distances[res_idx] = float(np.min(distances))
                
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
    print()
    
    # Process each file
    successful = 0
    failed = 0
    
    for cif_path in tqdm(cif_files, desc="Computing distances"):
        out_path = compute_distances_for_cif(str(cif_path), output_dir)
        if out_path:
            successful += 1
        else:
            failed += 1
    
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
