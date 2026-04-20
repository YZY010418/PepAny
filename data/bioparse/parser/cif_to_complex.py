#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
mmCIF parser for parsing CIF files from PDB or custom sources.
Supports extracting structures and atom-level filtering by ligand_ids.
"""

import gzip
import re
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

from ..hierarchy import Complex, Molecule, Block, Atom, Bond, BondType

try:
    from biotite.structure.io import pdbx
    from biotite.structure import BondType as BiotiteBondType
    _BIOTITE_AVAILABLE = True
except ImportError:
    pdbx = None
    BiotiteBondType = None
    _BIOTITE_AVAILABLE = False


def _bond_type_from_cif_order(order_str: Optional[str]):
    if order_str is None:
        return None
    order = str(order_str).strip().upper()
    mapping = {
        'SING': 1,
        'SINGLE': 1,
        'DOUB': 2,
        'DOUBLE': 2,
        'TRIP': 3,
        'TRIPLE': 3,
        'AROM': 4,
        'AROMATIC': 4,
    }
    value = mapping.get(order, None)
    return None if value is None else value


def _bond_type_from_biotite(bond):
    if BiotiteBondType is None:
        return BondType.NONE
    biotite_bond = bond if isinstance(bond, BiotiteBondType) else BiotiteBondType(bond)
    return BondType(biotite_bond.without_aromaticity().value)


def _normalize_ins_code(ins_code: Optional[str]) -> str:
    if ins_code in (None, '.', '?'):
        return ''
    return str(ins_code).strip()


def _parse_ligand_ids(ligand_ids_str: Optional[str]) -> Set[int]:
    if ligand_ids_str in (None, '.', '?'):
        return set()
    ligand_ids = set()
    for token in str(ligand_ids_str).split(','):
        token = token.strip()
        if not token:
            continue
        ligand_ids.add(int(token))
    return ligand_ids


def _get_residue_ligand_map(
    block_data: Dict[str, List[str]],
) -> Dict[Tuple[str, int, str], Set[int]]:
    """
    Build a residue-level ligand mapping from the CPSEA receptor-ligand loop.
    """
    chain_ids = block_data.get('_cpsea_receptor_ligand_map.auth_asym_id', [])
    seq_ids = block_data.get('_cpsea_receptor_ligand_map.auth_seq_id', [])
    ins_codes = block_data.get('_cpsea_receptor_ligand_map.pdbx_PDB_ins_code', [])
    label_chain_ids = block_data.get('_cpsea_receptor_ligand_map.label_asym_id', [])
    label_seq_ids = block_data.get('_cpsea_receptor_ligand_map.label_seq_id', [])
    ligand_ids = block_data.get('_cpsea_receptor_ligand_map.ligand_ids', [])

    if not (chain_ids and seq_ids and ligand_ids):
        return {}

    if not ins_codes:
        ins_codes = [''] * len(chain_ids)
    if not label_chain_ids:
        label_chain_ids = [''] * len(chain_ids)
    if not label_seq_ids:
        label_seq_ids = [''] * len(chain_ids)

    residue_ligand_map: Dict[Tuple[str, int, str], Set[int]] = {}
    for chain_id, seq_id, ins_code, label_chain_id, label_seq_id, ligand_ids_str in zip(
        chain_ids, seq_ids, ins_codes, label_chain_ids, label_seq_ids, ligand_ids
    ):
        try:
            ligand_id_set = _parse_ligand_ids(ligand_ids_str)
            residue_ligand_map[(chain_id, int(seq_id), _normalize_ins_code(ins_code))] = ligand_id_set
            if label_chain_id not in ('', '.', '?') and label_seq_id not in ('', '.', '?'):
                residue_ligand_map[(
                    label_chain_id,
                    int(label_seq_id),
                    _normalize_ins_code(ins_code)
                )] = ligand_id_set
        except (TypeError, ValueError):
            continue

    return residue_ligand_map


def _read_cif_file_with_biotite(filepath: str):
    if not _BIOTITE_AVAILABLE:
        return None
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return pdbx.CIFFile.read(f)
    return pdbx.CIFFile.read(filepath)


def _get_biotite_block_names(cif_file) -> List[str]:
    return list(cif_file.keys())


def _get_biotite_column(block, category_name: str, column_name: str):
    try:
        return block[category_name][column_name].as_array().tolist()
    except KeyError:
        return []


def _get_residue_ligand_map_from_biotite_block(block) -> Dict[Tuple[str, int, str], Set[int]]:
    block_data = {
        '_cpsea_receptor_ligand_map.auth_asym_id': _get_biotite_column(block, 'cpsea_receptor_ligand_map', 'auth_asym_id'),
        '_cpsea_receptor_ligand_map.auth_seq_id': _get_biotite_column(block, 'cpsea_receptor_ligand_map', 'auth_seq_id'),
        '_cpsea_receptor_ligand_map.pdbx_PDB_ins_code': _get_biotite_column(block, 'cpsea_receptor_ligand_map', 'pdbx_PDB_ins_code'),
        '_cpsea_receptor_ligand_map.label_asym_id': _get_biotite_column(block, 'cpsea_receptor_ligand_map', 'label_asym_id'),
        '_cpsea_receptor_ligand_map.label_seq_id': _get_biotite_column(block, 'cpsea_receptor_ligand_map', 'label_seq_id'),
        '_cpsea_receptor_ligand_map.ligand_ids': _get_biotite_column(block, 'cpsea_receptor_ligand_map', 'ligand_ids'),
    }
    return _get_residue_ligand_map(block_data)


def _find_ligand_block_name(block_names: List[str], ligand_id: int) -> Optional[str]:
    suffix = f'_ligand{ligand_id}'.lower()
    for block_name in block_names:
        lower_name = block_name.lower()
        if lower_name == f'ligand{ligand_id}' or lower_name.endswith(suffix):
            return block_name
    return None


def _cif_to_complex_with_biotite(
    filepath: str,
    selected_chains: Optional[List[str]] = None,
    block_name: Optional[str] = None,
    ligand_ids_filter: Optional[Set[int]] = None,
) -> Complex:
    cif_file = _read_cif_file_with_biotite(filepath)
    block_names = _get_biotite_block_names(cif_file)
    if len(block_names) == 0:
        raise ValueError(f"No data blocks found in {filepath}")

    if block_name is None:
        block_name = block_names[0]
    elif block_name not in block_names:
        raise ValueError(f"Block '{block_name}' not found in {filepath}")

    block = cif_file[block_name]
    struct = pdbx.get_structure(
        cif_file,
        model=1,
        data_block=block_name,
        use_author_fields=True,
        include_bonds=True,
        extra_fields=['atom_id']
    )

    residue_ligand_map = _get_residue_ligand_map_from_biotite_block(block)
    atom_site_ligand_ids = _get_biotite_column(block, 'atom_site', 'ligand_ids')

    mask = np.ones(struct.array_length(), dtype=bool)

    if selected_chains is not None:
        selected_chains = {str(chain_id) for chain_id in selected_chains}
        mask &= np.array([str(chain_id) in selected_chains for chain_id in struct.chain_id], dtype=bool)

    if ligand_ids_filter is not None:
        ligand_mask = np.zeros(struct.array_length(), dtype=bool)
        for i in range(struct.array_length()):
            residue_key = (
                str(struct.chain_id[i]),
                int(struct.res_id[i]),
                _normalize_ins_code(str(struct.ins_code[i]))
            )
            if residue_ligand_map:
                ligand_mask[i] = bool(residue_ligand_map.get(residue_key, set()) & ligand_ids_filter)
            elif i < len(atom_site_ligand_ids):
                ligand_mask[i] = bool(_parse_ligand_ids(atom_site_ligand_ids[i]) & ligand_ids_filter)
            else:
                ligand_mask[i] = True
        mask &= ligand_mask

    struct = struct[mask]

    molecules = []
    bonds = []
    residue_atoms = {}
    residue_names = {}
    chain_residues = {}

    atom_id_annotation = getattr(struct, 'atom_id', None)

    for i in range(struct.array_length()):
        atom = struct[i]
        chain_id = str(atom.chain_id)
        res_name = str(atom.res_name)
        res_number = int(atom.res_id)
        insert_code = _normalize_ins_code(str(atom.ins_code))
        res_id = (res_number, insert_code)
        atom_id = str(atom_id_annotation[i]) if atom_id_annotation is not None else str(i + 1)
        element = str(atom.element)

        if element == 'H':
            continue

        atom_instance = Atom(
            name=str(atom.atom_name).strip(),
            coordinate=atom.coord.tolist(),
            element=element,
            id=atom_id,
            properties={}
        )

        if (chain_id, res_id) not in residue_atoms:
            residue_atoms[(chain_id, res_id)] = []
            residue_names[(chain_id, res_id)] = res_name
        residue_atoms[(chain_id, res_id)].append((atom_instance, i))

    for (chain_id, res_id), atoms in residue_atoms.items():
        block = Block(
            name=residue_names[(chain_id, res_id)],
            atoms=[atom for atom, _ in atoms],
            id=res_id,
            properties={}
        )
        chain_residues.setdefault(chain_id, []).append(block)

    for chain_id in sorted(chain_residues.keys()):
        blocks = sorted(chain_residues[chain_id], key=lambda block: block.id)
        molecules.append(Molecule(name=chain_id, id=chain_id, blocks=blocks, properties={}))

    atom_to_molecule_block_atom = {}
    for mol_idx, molecule in enumerate(molecules):
        for block_idx, block in enumerate(molecule.blocks):
            residue_key = (molecule.id, block.id)
            for atom_idx, (_, source_idx) in enumerate(residue_atoms[residue_key]):
                atom_to_molecule_block_atom[source_idx] = (mol_idx, block_idx, atom_idx)

    if struct.bonds is not None:
        end_atoms, bond_types = struct.bonds.get_all_bonds()
        for begin_idx in range(len(end_atoms)):
            for end_idx, bond_type in zip(end_atoms[begin_idx], bond_types[begin_idx]):
                if end_idx < 0 or end_idx <= begin_idx:
                    continue
                if begin_idx not in atom_to_molecule_block_atom or end_idx not in atom_to_molecule_block_atom:
                    continue
                bonds.append(Bond(
                    index1=atom_to_molecule_block_atom[begin_idx],
                    index2=atom_to_molecule_block_atom[end_idx],
                    bond_type=_bond_type_from_biotite(bond_type)
                ))

    if struct.array_length() > 0 and len(bonds) == 0:
        print(f"Warning: No bond info parsed from {filepath} block '{block_name}'")

    return Complex(
        name=block_name,
        molecules=molecules,
        bonds=bonds,
        properties={}
    )


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
    if _BIOTITE_AVAILABLE:
        return _cif_to_complex_with_biotite(
            filepath,
            selected_chains=selected_chains,
            block_name=block_name,
            ligand_ids_filter=ligand_ids_filter,
        )

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
    atom_site_ids = block_data.get('_atom_site.id', [])
    auth_seq_ids = block_data.get('_atom_site.auth_seq_id', [])
    pdbx_ins_codes = block_data.get('_atom_site.pdbx_PDB_ins_code', [''] * len(atom_ids))
    cartn_x = block_data.get('_atom_site.Cartn_x', [])
    cartn_y = block_data.get('_atom_site.Cartn_y', [])
    cartn_z = block_data.get('_atom_site.Cartn_z', [])
    type_symbols = block_data.get('_atom_site.type_symbol', [])
    ligand_ids_str = block_data.get('_atom_site.ligand_ids', [])  # comma-separated string
    residue_ligand_map = _get_residue_ligand_map(block_data)
    
    # Filter by ligand_ids if specified. Prefer the residue-level CPSEA mapping,
    # and fall back to the older per-atom annotation when present.
    atoms_to_include = []
    for i in range(len(atom_ids)):
        if ligand_ids_filter is None:
            atoms_to_include.append(i)
            continue

        try:
            residue_key = (
                auth_asym_ids[i],
                int(auth_seq_ids[i]),
                _normalize_ins_code(pdbx_ins_codes[i]),
            )
        except (IndexError, TypeError, ValueError):
            residue_key = None

        if residue_key is not None and residue_ligand_map:
            if residue_ligand_map.get(residue_key, set()) & ligand_ids_filter:
                atoms_to_include.append(i)
            continue

        if i < len(ligand_ids_str):
            try:
                if _parse_ligand_ids(ligand_ids_str[i]) & ligand_ids_filter:
                    atoms_to_include.append(i)
            except (TypeError, ValueError):
                atoms_to_include.append(i)
        else:
            # Keep atoms if no ligand annotation is available for filtering.
            atoms_to_include.append(i)
    
    # Build molecule dictionary
    molecules_dict = defaultdict(lambda: defaultdict(list))
    
    for idx in atoms_to_include:
        chain_id = auth_asym_ids[idx]
        
        # Filter chains if specified
        if selected_chains and chain_id not in selected_chains:
            continue
        
        try:
            residue_num = int(auth_seq_ids[idx])
        except (TypeError, ValueError, IndexError):
            continue
        insertion_code = _normalize_ins_code(pdbx_ins_codes[idx] if idx < len(pdbx_ins_codes) else '')
        residue_name = auth_comp_ids[idx]
        atom_name = auth_atom_ids[idx]
        element = type_symbols[idx].strip()
        atom_id = atom_site_ids[idx] if idx < len(atom_site_ids) else str(idx + 1)

        if element == 'H':
            continue
        
        # Parse coordinates
        try:
            x, y, z = float(cartn_x[idx]), float(cartn_y[idx]), float(cartn_z[idx])
        except (ValueError, IndexError):
            continue
        
        # Create atom
        atom = Atom(
            name=atom_name,
            element=element,
            coordinate=[x, y, z],
            id=str(atom_id),
            properties={}
        )
        
        molecules_dict[chain_id][(residue_num, insertion_code)].append((residue_name, atom))
    
    # Build Molecule and Block objects
    molecules = []
    for chain_id in sorted(molecules_dict.keys()):
        blocks = []
        for residue_id in sorted(molecules_dict[chain_id].keys()):
            residue_atoms = molecules_dict[chain_id][residue_id]
            if residue_atoms:
                residue_name = residue_atoms[0][0]
                atoms_list = [atom for _, atom in residue_atoms]
                
                block = Block(
                    name=residue_name,
                    id=residue_id,
                    atoms=atoms_list,
                    properties={}
                )
                blocks.append(block)
        
        if blocks:
            molecule = Molecule(
                name=chain_id,
                id=chain_id,
                blocks=blocks,
                properties={}
            )
            molecules.append(molecule)

    # Build residue/atom lookup for bond parsing.
    residue_atom_lookup = {}
    for mol_idx, molecule in enumerate(molecules):
        for block_idx, block in enumerate(molecule.blocks):
            atom_name_to_idx = {atom.name: atom_idx for atom_idx, atom in enumerate(block.atoms)}
            residue_atom_lookup[(molecule.id, block.id)] = (block.name, atom_name_to_idx, mol_idx, block_idx)

    # Extract bonds if present
    bonds = []
    bond_keys = set()

    def _append_bond(index1, index2, bond_type_value):
        if bond_type_value is None:
            return
        key = (index1, index2) if index1 <= index2 else (index2, index1)
        if key in bond_keys:
            return
        bond_keys.add(key)
        bonds.append(Bond(index1=index1, index2=index2, bond_type=BondType(bond_type_value)))

    if '_chem_comp_bond.atom_id_1' in block_data:
        atom_id_1s = block_data['_chem_comp_bond.atom_id_1']
        atom_id_2s = block_data['_chem_comp_bond.atom_id_2']
        comp_ids = block_data.get('_chem_comp_bond.comp_id', [''] * len(atom_id_1s))
        bond_orders = block_data.get('_chem_comp_bond.value_order', ['SING'] * len(atom_id_1s))

        for comp_id, atom1, atom2, order_str in zip(comp_ids, atom_id_1s, atom_id_2s, bond_orders):
            bond_type_value = _bond_type_from_cif_order(order_str)
            for (chain_id, residue_id), (block_name, atom_name_to_idx, mol_idx, block_idx) in residue_atom_lookup.items():
                if block_name != comp_id:
                    continue
                if atom1 not in atom_name_to_idx or atom2 not in atom_name_to_idx:
                    continue
                index1 = (mol_idx, block_idx, atom_name_to_idx[atom1])
                index2 = (mol_idx, block_idx, atom_name_to_idx[atom2])
                _append_bond(index1, index2, bond_type_value)

    if '_struct_conn.ptnr1_label_atom_id' in block_data:
        chain_ids_1 = block_data.get('_struct_conn.ptnr1_label_asym_id', [])
        chain_ids_2 = block_data.get('_struct_conn.ptnr2_label_asym_id', [])
        seq_ids_1 = block_data.get('_struct_conn.ptnr1_label_seq_id', [])
        seq_ids_2 = block_data.get('_struct_conn.ptnr2_label_seq_id', [])
        atom_ids_1 = block_data.get('_struct_conn.ptnr1_label_atom_id', [])
        atom_ids_2 = block_data.get('_struct_conn.ptnr2_label_atom_id', [])
        ins_codes_1 = block_data.get('_struct_conn.pdbx_ptnr1_PDB_ins_code', [''] * len(atom_ids_1))
        ins_codes_2 = block_data.get('_struct_conn.pdbx_ptnr2_PDB_ins_code', [''] * len(atom_ids_2))
        bond_orders = block_data.get('_struct_conn.pdbx_value_order', ['SING'] * len(atom_ids_1))

        for chain1, chain2, seq1, seq2, atom1, atom2, ins1, ins2, order_str in zip(
            chain_ids_1, chain_ids_2, seq_ids_1, seq_ids_2, atom_ids_1, atom_ids_2, ins_codes_1, ins_codes_2, bond_orders
        ):
            try:
                residue_id_1 = (int(seq1), _normalize_ins_code(ins1))
                residue_id_2 = (int(seq2), _normalize_ins_code(ins2))
            except (TypeError, ValueError):
                continue
            entry1 = residue_atom_lookup.get((chain1, residue_id_1), None)
            entry2 = residue_atom_lookup.get((chain2, residue_id_2), None)
            if entry1 is None or entry2 is None:
                continue
            _, atom_name_to_idx_1, mol_idx_1, block_idx_1 = entry1
            _, atom_name_to_idx_2, mol_idx_2, block_idx_2 = entry2
            if atom1 not in atom_name_to_idx_1 or atom2 not in atom_name_to_idx_2:
                continue
            bond_type_value = _bond_type_from_cif_order(order_str)
            index1 = (mol_idx_1, block_idx_1, atom_name_to_idx_1[atom1])
            index2 = (mol_idx_2, block_idx_2, atom_name_to_idx_2[atom2])
            _append_bond(index1, index2, bond_type_value)

    if len(molecules) > 0 and len(bonds) == 0:
        print(f"Warning: No bond info parsed from {filepath} block '{block_name}'")
    
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
    if _BIOTITE_AVAILABLE:
        cif_file = _read_cif_file_with_biotite(filepath)
        ligand_block_name = _find_ligand_block_name(_get_biotite_block_names(cif_file), ligand_id)
        if ligand_block_name is None:
            raise ValueError(f"Ligand {ligand_id} not found in {filepath}")
        return cif_to_complex(filepath, block_name=ligand_block_name)

    data_blocks = parse_cif_file(filepath)
    
    # Find ligand block
    ligand_block_name = _find_ligand_block_name(list(data_blocks.keys()), ligand_id)
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
    if _BIOTITE_AVAILABLE:
        cif_file = _read_cif_file_with_biotite(filepath)
        block_names = _get_biotite_block_names(cif_file)
        if len(block_names) == 0:
            return set()
        if receptor_block_name is None:
            receptor_block_name = block_names[0]
        block = cif_file[receptor_block_name]
        residue_ligand_map = _get_residue_ligand_map_from_biotite_block(block)
        ligand_ids = set()
        if residue_ligand_map:
            for ids in residue_ligand_map.values():
                ligand_ids.update(ids)
            return ligand_ids
        for ligand_ids_str in _get_biotite_column(block, 'atom_site', 'ligand_ids'):
            try:
                ligand_ids.update(_parse_ligand_ids(ligand_ids_str))
            except (ValueError, TypeError):
                continue
        return ligand_ids

    data_blocks = parse_cif_file(filepath)
    
    if not data_blocks:
        return set()
    
    if receptor_block_name is None:
        # Find receptor block (usually first one)
        receptor_block_name = list(data_blocks.keys())[0]
    
    block_data = data_blocks[receptor_block_name]
    residue_ligand_map = _get_residue_ligand_map(block_data)

    ligand_ids = set()
    if residue_ligand_map:
        for ids in residue_ligand_map.values():
            ligand_ids.update(ids)
        return ligand_ids

    ligand_ids_str_list = block_data.get('_atom_site.ligand_ids', [])
    for ligand_ids_str in ligand_ids_str_list:
        try:
            ligand_ids.update(_parse_ligand_ids(ligand_ids_str))
        except (ValueError, TypeError):
            continue

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
