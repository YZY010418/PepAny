"""Accepted cyclic-peptide pipeline and terminal-cap utilities."""

import csv
import os
import re
try:
    import fcntl
except ImportError:
    fcntl = None
from copy import deepcopy
from pathlib import Path

from data.bioparse.hierarchy import Atom, Block, Bond, BondType, Complex, Molecule
from data.bioparse.utils import bond_type_from_rdkit
from data.bioparse.writer.complex_to_cif import complex_to_cif
from evaluation.terminal_amide_rmsd import (
    align_ref_to_generated,
    detect_clashes,
    detect_receptor_clashes,
    get_positions,
    heavy_atom_mol_with_map,
    infer_generated_indexes,
    kabsch_align,
    load_sdf_mol,
    merge_terminal_cap,
    remap_indexes,
    remove_atoms_with_map,
    rmsd,
    write_sdf,
)
from utils.logger import print_log

from .generation_common import _get_first


def _sdf_stem(path):
    path = Path(path)
    if path.name.endswith('.sdf'):
        return path.name[:-4]
    return path.stem


def _atom_name(symbol, counts):
    symbol = str(symbol).upper()
    counts[symbol] = counts.get(symbol, 0) + 1
    return f'{symbol}{counts[symbol]}'


def rdkit_atom_name(atom, counts):
    monomer_info = atom.GetMonomerInfo()
    if monomer_info is not None:
        name = monomer_info.GetName().strip()
        if name:
            return name
    if atom.HasProp('atom_name'):
        name = atom.GetProp('atom_name').strip()
        if name:
            return name
    return _atom_name(atom.GetSymbol(), counts)


def ligand_heavy_atom_refs(cplx, lig_chain):
    refs = []
    mol_idx = cplx.id2idx[lig_chain]
    for block_idx, block in enumerate(cplx[lig_chain]):
        for atom_idx, atom in enumerate(block):
            if str(atom.element).upper() in {'H', 'D'}:
                continue
            refs.append((mol_idx, block_idx, atom_idx))
    return refs


def remove_ligand_blocks_and_append_cap(cplx, lig_chain, remove_block_idxs, cap_block, cap_bonds, link_bonds):
    cplx = deepcopy(cplx)
    lig_mol_idx = cplx.id2idx[lig_chain]
    lig_mol = cplx[lig_chain]
    remove_block_idxs = set(int(idx) for idx in remove_block_idxs)
    cap_insert_idx = min(remove_block_idxs) if remove_block_idxs else len(lig_mol.blocks)

    kept_blocks = []
    block_old2new = {}
    for old_idx, block in enumerate(lig_mol.blocks):
        if old_idx == cap_insert_idx:
            cap_block_idx = len(kept_blocks)
            kept_blocks.append(cap_block)
        if old_idx in remove_block_idxs:
            continue
        block_old2new[old_idx] = len(kept_blocks)
        kept_blocks.append(block)

    if cap_insert_idx >= len(lig_mol.blocks):
        cap_block_idx = len(kept_blocks)
        kept_blocks.append(cap_block)

    molecules = []
    for mol_idx, mol in enumerate(cplx):
        if mol_idx == lig_mol_idx:
            molecules.append(Molecule(
                name=mol.name,
                blocks=kept_blocks,
                id=mol.id,
                properties=mol.properties,
            ))
        else:
            molecules.append(mol)

    bonds = []
    for bond in cplx.bonds:
        idx1 = tuple(bond.index1)
        idx2 = tuple(bond.index2)
        if idx1[0] == lig_mol_idx:
            if idx1[1] in remove_block_idxs:
                continue
            idx1 = (idx1[0], block_old2new[idx1[1]], idx1[2])
        if idx2[0] == lig_mol_idx:
            if idx2[1] in remove_block_idxs:
                continue
            idx2 = (idx2[0], block_old2new[idx2[1]], idx2[2])
        bonds.append(Bond(idx1, idx2, bond.bond_type))

    for atom_idx1, atom_idx2, bond_type in cap_bonds:
        bonds.append(Bond(
            index1=(lig_mol_idx, cap_block_idx, atom_idx1),
            index2=(lig_mol_idx, cap_block_idx, atom_idx2),
            bond_type=bond_type,
        ))

    for gen_ref, cap_atom_idx, bond_type in link_bonds:
        _mol_idx, block_idx, atom_idx = gen_ref
        bonds.append(Bond(
            index1=(lig_mol_idx, block_old2new[block_idx], atom_idx),
            index2=(lig_mol_idx, cap_block_idx, cap_atom_idx),
            bond_type=bond_type,
        ))

    return Complex(
        name=cplx.name,
        molecules=molecules,
        bonds=bonds,
        properties=cplx.properties,
    )


def merged_complex_with_cap_residue(cplx_desc, ref_mol, ref_indexes, gen_mol, gen_indexes, residue_name='CAP'):
    ref_mol, ref_old_to_new = heavy_atom_mol_with_map(ref_mol)
    gen_mol, gen_old_to_new = heavy_atom_mol_with_map(gen_mol)
    ref_indexes = remap_indexes(ref_indexes, ref_old_to_new, 'Reference')
    gen_indexes = remap_indexes(gen_indexes, gen_old_to_new, 'Generated')

    ref_to_gen_n = list(ref_indexes[:5])
    ref_to_gen_c = list(ref_indexes[5:])
    gen_n = list(gen_indexes[:5])
    gen_c = list(gen_indexes[5:])

    ref_work, gen_work = align_ref_to_generated(ref_mol, ref_indexes, gen_mol, gen_indexes)

    cplx = deepcopy(cplx_desc.cplx)
    lig_chain = cplx_desc.lig_chains[0]
    gen_refs = ligand_heavy_atom_refs(cplx, lig_chain)
    if len(gen_refs) != gen_work.GetNumAtoms():
        raise ValueError(
            f'Generated ligand atom count mismatch: complex has {len(gen_refs)} heavy atoms, '
            f'SDF has {gen_work.GetNumAtoms()}'
        )

    gen_conf = gen_work.GetConformer()
    for gen_idx, atom_ref in enumerate(gen_refs):
        pos = gen_conf.GetAtomPosition(gen_idx)
        cplx.get_atom(atom_ref).coordinate = [float(pos.x), float(pos.y), float(pos.z)]

    gen_remove = set(gen_n[:3] + gen_c[3:])
    ref_remove = set(ref_to_gen_n[3:] + ref_to_gen_c[:3])
    remove_block_idxs = {gen_refs[gen_idx][1] for gen_idx in gen_remove}

    ref_kept, ref_map = remove_atoms_with_map(ref_work, ref_remove)
    ref_conf = ref_kept.GetConformer()
    cap_name = str(residue_name)[:3]
    link_ref_atom_names = {
        ref_to_gen_n[1]: 'C',
        ref_to_gen_c[3]: 'N',
    }
    reserved_link_names = set(link_ref_atom_names.values())
    atom_counts = {}
    used_atom_names = set()
    cap_atoms = []
    for old_ref_idx, new_ref_idx in sorted(ref_map.items(), key=lambda item: item[1]):
        atom = ref_work.GetAtomWithIdx(old_ref_idx)
        pos = ref_conf.GetAtomPosition(new_ref_idx)
        if old_ref_idx in link_ref_atom_names:
            atom_name = link_ref_atom_names[old_ref_idx]
        else:
            atom_name = rdkit_atom_name(atom, atom_counts)
            if atom_name in reserved_link_names:
                atom_name = _atom_name(atom.GetSymbol(), atom_counts)
        while atom_name in used_atom_names:
            atom_name = _atom_name(atom.GetSymbol(), atom_counts)
        used_atom_names.add(atom_name)
        cap_atoms.append(Atom(
            name=atom_name,
            coordinate=[float(pos.x), float(pos.y), float(pos.z)],
            element=atom.GetSymbol(),
            id=str(new_ref_idx + 1),
        ))

    lig_mol = cplx[lig_chain]
    cap_res_id = lig_mol.blocks[min(remove_block_idxs)].id if remove_block_idxs else (1, '')
    cap_block = Block(
        name=cap_name,
        atoms=cap_atoms,
        id=cap_res_id,
        properties={'original_name': cap_name},
    )

    cap_bonds = []
    for bond in ref_kept.GetBonds():
        cap_bonds.append((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type_from_rdkit(bond),
        ))

    link_bonds = [
        (gen_refs[gen_n[3]], ref_map[ref_to_gen_n[1]], BondType.SINGLE),
        (gen_refs[gen_c[1]], ref_map[ref_to_gen_c[3]], BondType.SINGLE),
    ]
    return remove_ligand_blocks_and_append_cap(
        cplx,
        lig_chain,
        remove_block_idxs,
        cap_block,
        cap_bonds,
        link_bonds,
    )


def receptor_atoms_from_cplx(cplx, chain_ids):
    atoms = []
    for chain_id in chain_ids:
        if chain_id not in cplx.id2idx:
            continue
        for block in cplx[chain_id]:
            for atom in block:
                symbol = str(atom.element).upper()
                if symbol in {'H', 'D'}:
                    continue
                atoms.append((symbol, atom.coordinate))
    return atoms


def _zero_based_indexes(indexes, index_base):
    return [int(index) - int(index_base) for index in indexes]


def _coord_distance(coord1, coord2):
    return sum((float(coord1[i]) - float(coord2[i])) ** 2 for i in range(3)) ** 0.5


def _atom_by_name(block, atom_name):
    atom_name = str(atom_name).strip()
    for atom in block:
        if str(atom.name).strip() == atom_name:
            return atom
    return None


def _format_ligand_amide_violation(violation):
    return (
        f'{violation["chain"]}:{violation["prev_block"]}->{violation["next_block"]} '
        f'C-N={violation["distance"]:.3f}A'
    )


def find_ligand_amide_bond_violations(cplx, lig_chains, max_distance):
    violations = []
    max_distance = float(max_distance)
    for lig_chain in lig_chains:
        if lig_chain not in cplx.id2idx:
            continue
        blocks = list(cplx[lig_chain])
        for prev_block, next_block in zip(blocks[:-1], blocks[1:]):
            c_atom = _atom_by_name(prev_block, 'C')
            n_atom = _atom_by_name(next_block, 'N')
            if c_atom is None or n_atom is None:
                continue
            distance = _coord_distance(c_atom.coordinate, n_atom.coordinate)
            if distance > max_distance:
                violations.append({
                    'chain': lig_chain,
                    'prev_block': f'{prev_block.name}{prev_block.id}',
                    'next_block': f'{next_block.name}{next_block.id}',
                    'distance': distance,
                    'max_distance': max_distance,
                })
    return violations


def _merge_amide_check_options(config, template):
    opt = dict(config.get('amide_bond_check', {}) or {})
    opt.update(dict(template.pop('amide_bond_check', {}) or {}))
    opt.setdefault('enabled', True)
    opt.setdefault('max_c_n_distance', opt.get('max_distance', 1.8))
    return opt


def _amide_check_enabled(amide_check_opt):
    return bool(amide_check_opt.get('enabled', True))


def reject_implausible_ligand_amides(cplx_desc, details, n, amide_check_opt):
    if details is None or not _amide_check_enabled(amide_check_opt):
        return details
    max_distance = float(amide_check_opt.get('max_c_n_distance', amide_check_opt.get('max_distance', 1.8)))
    violations = find_ligand_amide_bond_violations(cplx_desc.cplx, cplx_desc.lig_chains, max_distance)
    if not violations:
        return details
    joined = '; '.join(_format_ligand_amide_violation(item) for item in violations[:3])
    if len(violations) > 3:
        joined += f'; +{len(violations) - 3} more'
    return _pipeline_reject(f'implausible ligand amide bond(s): {joined}', cplx_desc, n)


def _pipeline_reject(reason, cplx_desc, n):
    print_log(f'Rejected {cplx_desc.id}/{n}: {reason}', level='DEBUG')
    return None


def compute_terminal_rmsd_details(cplx_desc, raw_sdf_path, pipeline_opt):
    control = cplx_desc.control or {}
    ref_path = control.get('term_rel_pos_sdf_path', None)
    ref_atom_indices = control.get('term_rel_pos_atom_indices', None)
    if ref_path is None or ref_atom_indices is None:
        return {}

    ref_indexes = _zero_based_indexes(
        ref_atom_indices,
        control.get('term_rel_pos_index_base', 0),
    )
    ref_mol = load_sdf_mol(ref_path, control.get('term_rel_pos_mol_index', 0))
    gen_mol = load_sdf_mol(
        raw_sdf_path,
        int(_get_first(pipeline_opt, ('mol_index', 'gen_mol_index'), 0)),
    )
    gen_atom_indices = control.get('gen_atom_indices', pipeline_opt.get('gen_atom_indices', None))
    if gen_atom_indices is None:
        gen_indexes = infer_generated_indexes(gen_mol)
    else:
        gen_indexes = _zero_based_indexes(
            gen_atom_indices,
            control.get('gen_index_base', pipeline_opt.get('gen_index_base', 0)),
        )

    ref_coords = get_positions(ref_mol, ref_indexes)
    gen_coords = get_positions(gen_mol, gen_indexes)
    aligned = kabsch_align(gen_coords, ref_coords)
    terminal_rmsd = rmsd(aligned, ref_coords)
    return {
        'rmsd': terminal_rmsd,
        'terminal_rmsd': terminal_rmsd,
        'n_term_rmsd': rmsd(aligned[:5], ref_coords[:5]),
        'c_term_rmsd': rmsd(aligned[5:], ref_coords[5:]),
        'reference_sdf': str(ref_path),
        'conformer': _sdf_stem(ref_path),
    }


def add_terminal_rmsd_details(cplx_desc, details, raw_sdf_path, pipeline_opt):
    try:
        details.update(compute_terminal_rmsd_details(cplx_desc, raw_sdf_path, pipeline_opt))
    except Exception as exc:
        details['rmsd_error'] = str(exc)
    return details


PIPELINE_METADATA_FIELDS = [
    'accepted_name',
    'accepted_cif',
    'id',
    'sample_slot',
    'conformer',
    'rmsd',
    'reference_sdf',
    'terminal_rmsd',
    'n_term_rmsd',
    'c_term_rmsd',
    'rmsd_cutoff',
    'self_clash_count',
    'receptor_clash_count',
    'raw_pdb',
    'raw_sdf',
    'gen_seq',
    'smiles',
]


def append_pipeline_metadata(accepted_dir, row):
    path = Path(accepted_dir) / 'metadata.csv'
    with path.open('a+', newline='') as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        write_header = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_METADATA_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in PIPELINE_METADATA_FIELDS})
        handle.flush()
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def resolve_pipeline_accepted_dir(save_dir, pipeline_opt):
    output_dir = pipeline_opt.get('output_dir', None)
    if output_dir is None:
        return os.path.join(os.path.dirname(save_dir), 'accepted')
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(os.path.dirname(save_dir), output_dir)


def _pipeline_output_dir_name(pipeline_opt):
    output_dir = pipeline_opt.get('output_dir', None)
    if output_dir is None or os.path.isabs(output_dir):
        return 'accepted'
    return output_dir


def _read_pipeline_metadata(accepted_dir):
    metadata_path = Path(accepted_dir) / 'metadata.csv'
    if not metadata_path.exists():
        return []
    with metadata_path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _accepted_index(name):
    match = re.search(r'accepted_(\d+)$', str(name or ''))
    return None if match is None else int(match.group(1))


def run_terminal_merge_pipeline(
        cplx_desc,
        details,
        n,
        raw_pdb_path,
        raw_sdf_path,
        accepted_dir,
        pipeline_opt,
        accepted_index=None,
        accepted_name=None,
        append_metadata_row=True,
    ):
    control = cplx_desc.control or {}
    rmsd_details = compute_terminal_rmsd_details(cplx_desc, raw_sdf_path, pipeline_opt)
    ref_path = rmsd_details.get('reference_sdf', None)
    if ref_path is None:
        raise ValueError('pipeline.enabled requires term_rel_pos_sdf so each sample has a reference SDF')

    ref_atom_indices = control.get('term_rel_pos_atom_indices', None)
    ref_index_base = control.get('term_rel_pos_index_base', 0)
    ref_mol_index = control.get('term_rel_pos_mol_index', 0)
    gen_mol_index = int(_get_first(pipeline_opt, ('mol_index', 'gen_mol_index'), 0))
    ref_indexes = _zero_based_indexes(ref_atom_indices, ref_index_base)

    ref_mol = load_sdf_mol(ref_path, ref_mol_index)
    gen_mol = load_sdf_mol(raw_sdf_path, gen_mol_index)
    gen_atom_indices = control.get('gen_atom_indices', pipeline_opt.get('gen_atom_indices', None))
    if gen_atom_indices is None:
        gen_indexes = infer_generated_indexes(gen_mol)
    else:
        gen_indexes = _zero_based_indexes(
            gen_atom_indices,
            control.get('gen_index_base', pipeline_opt.get('gen_index_base', 0)),
        )

    ref_coords = get_positions(ref_mol, ref_indexes)
    gen_coords = get_positions(gen_mol, gen_indexes)
    aligned = kabsch_align(gen_coords, ref_coords)
    terminal_rmsd = rmsd(aligned, ref_coords)
    n_term_rmsd = rmsd(aligned[:5], ref_coords[:5])
    c_term_rmsd = rmsd(aligned[5:], ref_coords[5:])
    rmsd_cutoff = float(_get_first(pipeline_opt, ('rmsd_cutoff', 'merge_rmsd_cutoff'), 0.5))
    if terminal_rmsd >= rmsd_cutoff:
        return _pipeline_reject(f'terminal RMSD {terminal_rmsd:.4f} >= cutoff {rmsd_cutoff:.4f}', cplx_desc, n)

    merged_mol = merge_terminal_cap(ref_mol, ref_indexes, gen_mol, gen_indexes)
    clash_threshold = float(pipeline_opt.get('clash_threshold', 0.8))
    self_clashes = []
    receptor_clashes = []
    if not bool(pipeline_opt.get('no_clash_check', False)):
        self_clashes = detect_clashes(merged_mol, threshold=clash_threshold)
        receptor_atoms = receptor_atoms_from_cplx(cplx_desc.cplx, cplx_desc.tgt_chains)
        receptor_clashes = detect_receptor_clashes(merged_mol, receptor_atoms, threshold=clash_threshold)
        if self_clashes or receptor_clashes:
            return _pipeline_reject(
                f'clash self={len(self_clashes)} receptor={len(receptor_clashes)}',
                cplx_desc,
                n,
            )

    residue_name = pipeline_opt.get('ligand_residue_name', 'CAP')
    accepted_dir = Path(accepted_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    if accepted_name is None:
        accepted_name = f'accepted_{accepted_index}'
    cif_path = accepted_dir / f'{accepted_name}.cif'
    merged_cplx = merged_complex_with_cap_residue(
        cplx_desc,
        ref_mol,
        ref_indexes,
        gen_mol,
        gen_indexes,
        residue_name=residue_name,
    )
    complex_to_cif(
        merged_cplx,
        str(cif_path),
        selected_chains=cplx_desc.tgt_chains + cplx_desc.lig_chains,
        title=accepted_name,
    )

    merged_sdf_path = ''
    if bool(pipeline_opt.get('write_merged_sdf', False)):
        merged_sdf_path = str(accepted_dir / f'{accepted_name}.sdf')
        write_sdf(merged_mol, merged_sdf_path)

    details.update({
        'accepted_name': accepted_name,
        **rmsd_details,
        'rmsd_cutoff': rmsd_cutoff,
        'self_clash_count': len(self_clashes),
        'receptor_clash_count': len(receptor_clashes),
        'raw_pdb': str(raw_pdb_path),
        'raw_sdf': str(raw_sdf_path),
        'accepted_cif': str(cif_path),
        'merged_sdf': merged_sdf_path,
        'merge_status': 'saved',
    })
    if append_metadata_row:
        append_pipeline_metadata(accepted_dir, {
            **details,
            'id': cplx_desc.id,
            'sample_slot': n,
        })
    return details
