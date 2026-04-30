#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
from typing import List, Optional

from .. import const
from ..hierarchy import Complex, Block, BondType
from ..vocab import VOCAB


def _quote(value):
    value = '.' if value is None or value == '' else str(value)
    if re.fullmatch(r'[A-Za-z0-9_.+-]+', value):
        return value
    return "'" + value.replace("'", "''") + "'"


def _data_name(title, path):
    name = title if title is not None else str(path).split('/')[-1]
    name = re.sub(r'\.cif$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^A-Za-z0-9_]+', '_', name).strip('_')
    return name or 'complex'


def _block_name(block: Block):
    if 'original_name' in block.properties:
        return str(block.properties['original_name'])[:3]
    if block.name in const.AA_GEOMETRY:
        return block.name
    return VOCAB.abrv_to_symbol(block.name).replace('f', '')[-3:]


def complex_to_cif(
        cplx: Complex,
        cif_path: str,
        selected_chains: Optional[List[str]] = None,
        title: Optional[str] = None,
    ):
    """Write a Complex object as a minimal mmCIF coordinate file."""
    if selected_chains is None:
        selected_chains = [mol.id for mol in cplx]
    selected_chains = set(selected_chains)

    with open(cif_path, 'w') as fout:
        fout.write(f'data_{_data_name(title, cif_path)}\n#\n')
        fout.write('loop_\n')
        fout.write('_atom_site.group_PDB\n')
        fout.write('_atom_site.id\n')
        fout.write('_atom_site.type_symbol\n')
        fout.write('_atom_site.label_atom_id\n')
        fout.write('_atom_site.label_alt_id\n')
        fout.write('_atom_site.label_comp_id\n')
        fout.write('_atom_site.label_asym_id\n')
        fout.write('_atom_site.label_entity_id\n')
        fout.write('_atom_site.label_seq_id\n')
        fout.write('_atom_site.pdbx_PDB_ins_code\n')
        fout.write('_atom_site.Cartn_x\n')
        fout.write('_atom_site.Cartn_y\n')
        fout.write('_atom_site.Cartn_z\n')
        fout.write('_atom_site.occupancy\n')
        fout.write('_atom_site.B_iso_or_equiv\n')
        fout.write('_atom_site.auth_seq_id\n')
        fout.write('_atom_site.auth_comp_id\n')
        fout.write('_atom_site.auth_asym_id\n')
        fout.write('_atom_site.auth_atom_id\n')
        fout.write('_atom_site.pdbx_PDB_model_num\n')

        atom_number = 1
        entity_ids = {}
        atom_sites = {}
        for mol_idx, mol in enumerate(cplx):
            if mol.id not in selected_chains:
                continue
            entity_ids.setdefault(mol.id, len(entity_ids) + 1)
            for block_idx, block in enumerate(mol):
                block_name = _block_name(block)
                group = 'ATOM' if block.name in const.AA_GEOMETRY else 'HETATM'
                seq_id = int(block.id[0])
                insert_code = str(block.id[1]).strip() or '.'
                for atom_idx, atom in enumerate(block):
                    occupancy = atom.get_property('occupancy', 1.0)
                    bfactor = atom.get_property('bfactor', 0.0)
                    x, y, z = atom.coordinate
                    atom_sites[(mol_idx, block_idx, atom_idx)] = {
                        'chain': mol.id,
                        'block_name': block_name,
                        'block_is_standard': block.name in const.AA_GEOMETRY,
                        'seq_id': seq_id,
                        'atom_name': atom.name,
                        'coord': atom.coordinate,
                    }
                    row = [
                        group,
                        atom_number,
                        atom.element,
                        atom.name,
                        '.',
                        block_name,
                        mol.id,
                        entity_ids[mol.id],
                        seq_id,
                        insert_code,
                        f'{float(x):.3f}',
                        f'{float(y):.3f}',
                        f'{float(z):.3f}',
                        f'{float(occupancy):.2f}',
                        f'{float(bfactor):.2f}',
                        seq_id,
                        block_name,
                        mol.id,
                        atom.name,
                        1,
                    ]
                    fout.write(' '.join(_quote(value) for value in row) + '\n')
                    atom_number += 1
        fout.write('#\n')

        chem_bond_rows = []
        struct_conn_rows = []
        for bond_idx, bond in enumerate(cplx.bonds, start=1):
            site1 = atom_sites.get(tuple(bond.index1))
            site2 = atom_sites.get(tuple(bond.index2))
            if site1 is None or site2 is None:
                continue
            same_block = bond.index1[:2] == bond.index2[:2]
            if same_block and not site1['block_is_standard']:
                chem_bond_rows.append((bond_idx, site1, site2, bond.bond_type))
                continue
            if site1['block_is_standard'] and site2['block_is_standard']:
                continue
            struct_conn_rows.append((bond_idx, site1, site2, bond.bond_type))

        order_names = {
            BondType.SINGLE: 'SING',
            BondType.DOUBLE: 'DOUB',
            BondType.TRIPLE: 'TRIP',
            BondType.AROMATIC: 'AROM',
        }

        if chem_bond_rows:
            fout.write('loop_\n')
            fout.write('_chem_comp_bond.comp_id\n')
            fout.write('_chem_comp_bond.atom_id_1\n')
            fout.write('_chem_comp_bond.atom_id_2\n')
            fout.write('_chem_comp_bond.value_order\n')
            fout.write('_chem_comp_bond.pdbx_aromatic_flag\n')
            fout.write('_chem_comp_bond.pdbx_stereo_config\n')
            fout.write('_chem_comp_bond.pdbx_ordinal\n')

            for ordinal, (_bond_idx, site1, site2, bond_type) in enumerate(chem_bond_rows, start=1):
                row = [
                    site1['block_name'],
                    site1['atom_name'],
                    site2['atom_name'],
                    order_names.get(bond_type, 'SING'),
                    'Y' if bond_type == BondType.AROMATIC else 'N',
                    'N',
                    ordinal,
                ]
                fout.write(' '.join(_quote(value) for value in row) + '\n')
            fout.write('#\n')

        if struct_conn_rows:
            fout.write('loop_\n')
            fout.write('_struct_conn.id\n')
            fout.write('_struct_conn.conn_type_id\n')
            fout.write('_struct_conn.ptnr1_label_asym_id\n')
            fout.write('_struct_conn.ptnr1_label_comp_id\n')
            fout.write('_struct_conn.ptnr1_label_seq_id\n')
            fout.write('_struct_conn.ptnr1_label_atom_id\n')
            fout.write('_struct_conn.ptnr1_auth_asym_id\n')
            fout.write('_struct_conn.ptnr1_auth_comp_id\n')
            fout.write('_struct_conn.ptnr1_auth_seq_id\n')
            fout.write('_struct_conn.ptnr1_auth_atom_id\n')
            fout.write('_struct_conn.ptnr2_label_asym_id\n')
            fout.write('_struct_conn.ptnr2_label_comp_id\n')
            fout.write('_struct_conn.ptnr2_label_seq_id\n')
            fout.write('_struct_conn.ptnr2_label_atom_id\n')
            fout.write('_struct_conn.ptnr2_auth_asym_id\n')
            fout.write('_struct_conn.ptnr2_auth_comp_id\n')
            fout.write('_struct_conn.ptnr2_auth_seq_id\n')
            fout.write('_struct_conn.ptnr2_auth_atom_id\n')
            fout.write('_struct_conn.pdbx_value_order\n')
            fout.write('_struct_conn.pdbx_dist_value\n')

            for bond_idx, site1, site2, bond_type in struct_conn_rows:
                dist = sum(
                    (float(site1['coord'][i]) - float(site2['coord'][i])) ** 2
                    for i in range(3)
                ) ** 0.5
                row = [
                    f'covale{bond_idx}',
                    'covale',
                    site1['chain'],
                    site1['block_name'],
                    site1['seq_id'],
                    site1['atom_name'],
                    site1['chain'],
                    site1['block_name'],
                    site1['seq_id'],
                    site1['atom_name'],
                    site2['chain'],
                    site2['block_name'],
                    site2['seq_id'],
                    site2['atom_name'],
                    site2['chain'],
                    site2['block_name'],
                    site2['seq_id'],
                    site2['atom_name'],
                    order_names.get(bond_type, 'SING'),
                    f'{dist:.3f}',
                ]
                fout.write(' '.join(_quote(value) for value in row) + '\n')
            fout.write('#\n')
