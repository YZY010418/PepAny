#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
mmCIF parser for parsing CIF files from PDB or custom sources.
Supports extracting structures and atom-level filtering by ligand_ids.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from ..hierarchy import Complex, Molecule, Block, Atom, Bond


def parse_cif_file(filepath: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse mmCIF file and return dict of data blocks.
    
    Args:
        filepath: Path to .cif or .cif.gz file
        
    Returns:
        Dict mapping data block name to loop data
        e.g., {
            'data_8DTM_cluster0_receptor': {
                '_atom_site.label_atom_id': ['N', 'CA', ...],
                '_atom_site.Cartn_x': ['117.704', '117.963', ...],
                ...
            }
        }
    """
    # Handle gzip compression
    if filepath.endswith('.gz'):
        import gzip
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            content = f.read()
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    
    data_blocks = {}
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Find data block header
        if line.startswith('data_'):
            block_name = line[5:].strip()
            data_blocks[block_name] = {}
            i += 1
            continue
        
        # Find loop blocks
        if line.startswith('loop_'):
            # Collect column names
            column_names = []
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('_'):
                    column_names.append(line)
                    i += 1
                else:
                    break
            
            # Collect data values
            values = []
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith('#') or line.startswith('_') or line.startswith('loop_') or line.startswith('data_'):
                    break
                # Handle quoted strings
                tokens = re.findall(r"'[^']*'|\"[^\"]*\"|[^\s]+", line)
                values.extend([t.strip("'\"") for t in tokens])
                i += 1
            
            # Assign values to columns
            if len(values) % len(column_names) == 0:
                n_rows = len(values) // len(column_names)
                for col_idx, col_name in enumerate(column_names):
                    if block_name not in data_blocks:
                        data_blocks[block_name] = {}
                    data_blocks[block_name][col_name] = [
                        values[row_idx * len(column_names) + col_idx]
                        for row_idx in range(n_rows)
                    ]
            continue
        
        i += 1
    
    return data_blocks


def cif_to_complex(
    filepath: str,
    selected_chains: Optional[List[str]] = None,
    block_name: Optional[str] = None,
    ligand_ids_filter: Optional[Set[int]] = None,
) -> Complex:
    """
    Convert mmCIF file to Complex object.
    
    Args:
        filepath: Path to .cif or .cif.gz file
        selected_chains: List of chain IDs to extract (e.g., ['A', 'B']). 
                        If None, extract all chains.
        block_name: Specific data block to extract. If None, use first block.
        ligand_ids_filter: If provided, only include atoms whose ligand_ids 
                          contain at least one value in this set.
                          
    Returns:
        Complex object
    """
    data_blocks = parse_cif_file(filepath)
    
    if not data_blocks:
        raise ValueError(f"No data blocks found in {filepath}")
    
    # Select block to parse
    if block_name is None:
        block_name = list(data_blocks.keys())[0]
    elif block_name not in data_blocks:
        raise ValueError(f"Block '{block_name}' not found in {filepath}")
    
    block_data = data_blocks[block_name]
    
    # Parse atom sites
    if '_atom_site.label_atom_id' not in block_data:
        raise ValueError("_atom_site section not found in CIF file")
    
    atom_ids = block_data.get('_atom_site.label_atom_id', [])
    auth_atom_ids = block_data.get('_atom_site.auth_atom_id', atom_ids)
    auth_comp_ids = block_data.get('_atom_site.auth_comp_id', [])
    auth_asym_ids = block_data.get('_atom_site.auth_asym_id', [])  # chain ID
    auth_seq_ids = block_data.get('_atom_site.auth_seq_id', [])
    cartn_x = block_data.get('_atom_site.Cartn_x', [])
    cartn_y = block_data.get('_atom_site.Cartn_y', [])
    cartn_z = block_data.get('_atom_site.Cartn_z', [])
    type_symbols = block_data.get('_atom_site.type_symbol', [])
    ligand_ids_str = block_data.get('_atom_site.ligand_ids', [])  # comma-separated string
    
    # Filter by ligand_ids if specified
    atoms_to_include = []
    for i, ligand_ids_str_val in enumerate(ligand_ids_str):
        if ligand_ids_filter is None:
            atoms_to_include.append(i)
        else:
            # Parse comma-separated ligand IDs
            try:
                ligand_ids_set = set(int(x.strip()) for x in ligand_ids_str_val.split(','))
                if ligand_ids_set & ligand_ids_filter:  # intersection
                    atoms_to_include.append(i)
            except (ValueError, AttributeError):
                # If parsing fails, include atom
                atoms_to_include.append(i)
    
    # Build molecule dictionary
    molecules_dict = defaultdict(lambda: defaultdict(list))
    
    for idx in atoms_to_include:
        chain_id = auth_asym_ids[idx]
        
        # Filter chains if specified
        if selected_chains and chain_id not in selected_chains:
            continue
        
        residue_num = int(auth_seq_ids[idx])
        residue_name = auth_comp_ids[idx]
        atom_name = auth_atom_ids[idx]
        element = type_symbols[idx].strip()
        
        # Parse coordinates
        try:
            x, y, z = float(cartn_x[idx]), float(cartn_y[idx]), float(cartn_z[idx])
        except (ValueError, IndexError):
            continue
        
        # Create atom
        atom = Atom(
            name=atom_name,
            element=element,
            coord=[x, y, z],
            properties={}
        )
        
        molecules_dict[chain_id][residue_num].append((residue_name, atom))
    
    # Build Molecule and Block objects
    molecules = []
    for chain_id in sorted(molecules_dict.keys()):
        blocks = []
        for residue_num in sorted(molecules_dict[chain_id].keys()):
            residue_atoms = molecules_dict[chain_id][residue_num]
            if residue_atoms:
                residue_name = residue_atoms[0][0]
                atoms_list = [atom for _, atom in residue_atoms]
                
                block = Block(
                    name=residue_name,
                    id=(residue_num, ''),  # (sequence number, insertion code)
                    atoms=atoms_list,
                    properties={}
                )
                blocks.append(block)
        
        if blocks:
            molecule = Molecule(
                id=chain_id,
                blocks=blocks,
                properties={}
            )
            molecules.append(molecule)
    
    # Extract bonds if present
    bonds = []
    if '_chem_comp_bond.atom_id_1' in block_data:
        atom_id_1s = block_data['_chem_comp_bond.atom_id_1']
        atom_id_2s = block_data['_chem_comp_bond.atom_id_2']
        bond_orders = block_data.get('_chem_comp_bond.value_order', ['SING'] * len(atom_id_1s))
        
        # Map bond order strings to values
        bond_order_map = {'SING': 1, 'DOUB': 2, 'TRIP': 3, 'AROM': 1.5}
        
        for atom1, atom2, order_str in zip(atom_id_1s, atom_id_2s, bond_orders):
            # For now, skip bonds (complex indexing needed)
            # This would require tracking atom indices across molecules/blocks
            pass
    
    complex_obj = Complex(
        name=block_name,
        molecules=molecules,
        bonds=bonds,
        properties={}
    )
    
    return complex_obj


def extract_ligand_from_cif(
    filepath: str,
    ligand_id: int,
) -> Complex:
    """
    Extract a specific ligand structure from mmCIF file.
    
    Args:
        filepath: Path to .cif or .cif.gz file
        ligand_id: Ligand ID to extract (1-indexed)
        
    Returns:
        Complex object containing only the ligand structure
    """
    data_blocks = parse_cif_file(filepath)
    
    # Find ligand block
    ligand_block_name = None
    for block_name in data_blocks.keys():
        if f'ligand{ligand_id}' in block_name.lower():
            ligand_block_name = block_name
            break
    
    if ligand_block_name is None:
        raise ValueError(f"Ligand {ligand_id} not found in {filepath}")
    
    return cif_to_complex(filepath, block_name=ligand_block_name)


def get_ligand_ids_from_receptor(
    filepath: str,
    receptor_block_name: Optional[str] = None,
) -> Set[int]:
    """
    Extract all unique ligand IDs from receptor structure.
    
    Args:
        filepath: Path to .cif or .cif.gz file
        receptor_block_name: Name of receptor block. If None, use first block.
        
    Returns:
        Set of ligand IDs (integers)
    """
    data_blocks = parse_cif_file(filepath)
    
    if not data_blocks:
        return set()
    
    if receptor_block_name is None:
        # Find receptor block (usually first one)
        receptor_block_name = list(data_blocks.keys())[0]
    
    block_data = data_blocks[receptor_block_name]
    ligand_ids_str_list = block_data.get('_atom_site.ligand_ids', [])
    
    ligand_ids = set()
    for ligand_ids_str in ligand_ids_str_list:
        try:
            ids = set(int(x.strip()) for x in ligand_ids_str.split(','))
            ligand_ids.update(ids)
        except (ValueError, AttributeError):
            pass
    
    return ligand_ids


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cif_to_complex.py <cif_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Parse all blocks
    data_blocks = parse_cif_file(filepath)
    print(f"Found {len(data_blocks)} data blocks:")
    for block_name in data_blocks.keys():
        print(f"  - {block_name}")
    
    # Get ligand IDs from receptor
    ligand_ids = get_ligand_ids_from_receptor(filepath)
    print(f"Ligand IDs in receptor: {sorted(ligand_ids)}")
    
    # Parse receptor
    try:
        receptor = cif_to_complex(filepath)
        print(f"Receptor: {len(receptor.molecules)} chains, {sum(len(m.blocks) for m in receptor.molecules)} residues")
    except Exception as e:
        print(f"Error parsing receptor: {e}")
