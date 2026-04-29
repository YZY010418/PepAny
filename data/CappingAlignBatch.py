#!/usr/bin/env python3
"""Batch-align relaxed SDF cap conformers to CPSea multi-ligand receptor CIF files.

The script reads base ``.cif``/``.cif.gz`` files that contain one receptor block
and multiple ligand blocks, aligns cap SDF conformers to every ligand terminal
amide pair, checks receptor clashes, and writes one capped ligand + receptor
PDB per match.
"""

import argparse
import csv
import gzip
import os
import re
import shutil
import shlex
import tempfile
from dataclasses import dataclass
from multiprocessing import Manager, Pool, cpu_count
from pathlib import Path

import numpy as np
import yaml
from sklearn.neighbors import BallTree

import biotite.structure as struc
from biotite.structure import superimpose

from CappingAlign import (
    cal_descriptor,
    clash_check,
    get_atoms_coords,
    read_structure,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

try:
    from rdkit import Chem
except ImportError:
    Chem = None


HEAVY_ELEMENTS = {"C", "N", "O", "S", "P", "F", "CL", "BR", "I"}
WORKER_ARGS = None
WORKER_CAP_INDEXES = None
WORKER_CAP_NUMBERING = None
WORKER_TMP_ROOT = None
WORKER_MERGED_COUNTER = None
WORKER_MERGED_COUNTER_LOCK = None
WORKER_METADATA_PATH = None
WORKER_METADATA_LOCK = None

METADATA_FIELDNAMES = [
    "cyclic_type_name",
    "merged_number",
    "base_name",
    "ligand_id",
    "cap_name",
    "mode",
    "descriptor_distance",
    "rmsd",
]


@dataclass
class TerminalDefinition:
    ref_coords: np.ndarray
    ligand_targets: list
    remove_res_ids: list
    cap_remove_atoms: list
    new_bonds: list
    cap_res_id: int


def strip_structure_suffix(path):
    name = Path(path).name
    for suffix in (".cif.gz", ".cif", ".sdf", ".sd", ".pdb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def sanitize_filename_component(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or "cyclic"


def existing_merged_max_number(output_dir, cyclic_type_name):
    prefix = sanitize_filename_component(cyclic_type_name)
    pattern = re.compile(rf"^{re.escape(prefix)}_merged_(\d+)\.pdb$")
    max_number = 0
    for path in Path(output_dir).iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        max_number = max(max_number, int(match.group(1)))
    return max_number


def next_merged_output_path(output_dir, cyclic_type_name):
    prefix = sanitize_filename_component(cyclic_type_name)
    WORKER_MERGED_COUNTER_LOCK.acquire()
    try:
        WORKER_MERGED_COUNTER.value += 1
        number = WORKER_MERGED_COUNTER.value
    finally:
        WORKER_MERGED_COUNTER_LOCK.release()
    return number, Path(output_dir) / f"{prefix}_merged_{number}.pdb"


def write_metadata_header(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()


def append_metadata_row(row):
    WORKER_METADATA_LOCK.acquire()
    try:
        with open(WORKER_METADATA_PATH, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDNAMES)
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDNAMES})
    finally:
        WORKER_METADATA_LOCK.release()


def read_text(path):
    path = str(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return handle.read()
    with open(path) as handle:
        return handle.read()


def split_cif_blocks(text):
    blocks = {}
    current_name = None
    current_lines = []

    for line in text.splitlines():
        if line.startswith("data_"):
            if current_name is not None:
                blocks[current_name] = "\n".join(current_lines)
            current_name = line[5:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_name is not None:
        blocks[current_name] = "\n".join(current_lines)

    return blocks


def tokenize_mmcif_line(line):
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def parse_loop(block_text, category_prefix):
    lines = block_text.splitlines()
    rows = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        tags = []
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if not stripped.startswith("_"):
                break
            tags.append(tokenize_mmcif_line(stripped)[0])
            i += 1

        data_tokens = []
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if stripped.startswith("#"):
                i += 1
                break
            if stripped == "loop_" or stripped.startswith("data_") or stripped.startswith("_"):
                break
            if stripped.startswith(";"):
                token_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith(";"):
                    token_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    i += 1
                data_tokens.append("\n".join(token_lines))
                continue
            data_tokens.extend(tokenize_mmcif_line(stripped))
            i += 1

        if not tags or not any(tag.startswith(category_prefix) for tag in tags):
            continue

        if len(data_tokens) % len(tags) != 0:
            raise ValueError(
                f"Cannot parse {category_prefix} loop: {len(data_tokens)} values "
                f"for {len(tags)} tags"
            )

        short_tags = [tag.split(".", 1)[1] if "." in tag else tag for tag in tags]
        for start in range(0, len(data_tokens), len(tags)):
            rows.append(dict(zip(short_tags, data_tokens[start : start + len(tags)])))

    return rows


def normalize_ins_code(value):
    if value in (None, "", " ", ".", "?"):
        return ""
    return str(value).strip()


def residue_keys_from_row(row):
    keys = set()
    label_asym = row.get("label_asym_id", "")
    label_seq = row.get("label_seq_id", "")
    auth_asym = row.get("auth_asym_id", "")
    auth_seq = row.get("auth_seq_id", "")
    ins = normalize_ins_code(row.get("pdbx_PDB_ins_code", ""))

    if label_asym and label_seq not in ("", ".", "?"):
        keys.add(("label", str(label_asym), str(label_seq), ins))
    if auth_asym and auth_seq not in ("", ".", "?"):
        keys.add(("auth", str(auth_asym), str(auth_seq), ins))
    return keys


def parse_receptor_ligand_map(receptor_block_text):
    rows = parse_loop(receptor_block_text, "_cpsea_receptor_ligand_map.")
    ligand_to_receptor = {}

    for row in rows:
        ligand_ids_raw = row.get("ligand_ids", "")
        keys = residue_keys_from_row(row)
        if not keys:
            continue
        for token in ligand_ids_raw.split(","):
            token = token.strip()
            if not token:
                continue
            ligand_to_receptor.setdefault(token, set()).update(keys)

    if not ligand_to_receptor:
        raise ValueError("No usable _cpsea_receptor_ligand_map rows found")

    return ligand_to_receptor


def atom_name_field(atom_name, element):
    atom_name = str(atom_name).strip()[:4]
    element = str(element).strip()
    if len(atom_name) >= 4:
        return atom_name
    if len(element) == 1 and atom_name and atom_name[0].isalpha():
        return f" {atom_name:<3}"
    return f"{atom_name:>4}"


def pdb_atom_line(serial, atom_name, res_name, chain_id, res_id, x, y, z, element, record="ATOM"):
    return (
        f"{record:<6}{serial:>5d} {atom_name_field(atom_name, element)} "
        f"{res_name[:3]:>3s} {chain_id[:1]:1s}{res_id:>4d}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}"
        f"          {element:>2s}\n"
    )


def bond_order_from_value(value):
    value = str(value or "").strip().upper()
    if value in {"DOUB", "DOUBLE", "2"}:
        return 2
    if value in {"TRIP", "TRIPLE", "3"}:
        return 3
    return 1


def bond_order_from_biotite_row(bond):
    if len(bond) < 3:
        return 1
    try:
        order = int(bond[2])
    except Exception:
        return 1
    if order in (2, 3, 4):
        return order
    return 1


def bond_order_from_rdkit_bond(bond):
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.BondType.AROMATIC:
        return 4
    return 1


def add_bond_order(bonds, serial1, serial2, order=1):
    key = tuple(sorted((int(serial1), int(serial2))))
    bonds[key] = max(int(order), bonds.get(key, 1))


def write_conect_records(handle, bonds):
    for (serial1, serial2), order in sorted(bonds.items()):
        try:
            order = int(order)
        except Exception:
            order = 1
        order = order if order in (2, 3) else 1
        repeated_target = "".join(f"{serial2:>5d}" for _ in range(order))
        handle.write(f"CONECT{serial1:>5d}{repeated_target}\n")


def parse_atom_names_property(mol):
    for prop_name in ("Atom_Names", "atom_names", "ATOM_NAMES"):
        if not mol.HasProp(prop_name):
            continue
        value = mol.GetProp(prop_name).strip()
        if not value:
            continue
        if "," in value:
            names = [item.strip() for item in value.split(",")]
        else:
            names = value.split()
        if len(names) == mol.GetNumAtoms():
            return {idx: {"atom_name": name} for idx, name in enumerate(names)}
    return {}


def atom_name_sidecar_candidates(path):
    path = Path(path)
    stem = strip_structure_suffix(path)
    candidates = [path.with_name(f"{stem}_atom_names.csv")]
    if "_conformers_" in stem:
        cap_name = stem.split("_conformers_", 1)[0]
        candidates.append(path.with_name(f"{cap_name}_atom_names.csv"))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        unique_candidates.append(candidate)
        seen.add(candidate)
    return unique_candidates


def load_atom_name_sidecar(path):
    candidates = [candidate for candidate in atom_name_sidecar_candidates(path) if candidate.exists()]
    if not candidates:
        maps = sorted(Path(path).parent.glob("*_atom_names.csv"))
        if len(maps) == 1:
            candidates = maps
    if not candidates:
        return {}

    metadata = {}
    with open(candidates[0], newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("atom_index", "") != "":
                atom_idx = int(row["atom_index"])
            elif row.get("atom_serial", "") != "":
                atom_idx = int(row["atom_serial"]) - 1
            else:
                continue
            metadata[atom_idx] = {
                "atom_name": row.get("atom_name", "").strip(),
                "chain_id": row.get("chain_id", "").strip(),
                "residue_name": row.get("residue_name", "").strip(),
                "residue_number": row.get("residue_number", "").strip(),
            }
    return metadata


def sdf_atom_metadata(path, mol):
    metadata = parse_atom_names_property(mol)
    sidecar_metadata = load_atom_name_sidecar(path)
    for atom_idx, values in sidecar_metadata.items():
        metadata.setdefault(atom_idx, {}).update({key: value for key, value in values.items() if value})
    return metadata


def rdkit_atom_name(atom, atom_counts, metadata=None):
    if metadata:
        name = metadata.get("atom_name", "").strip()
        if name:
            return name
    monomer_info = atom.GetMonomerInfo()
    if monomer_info is not None:
        name = monomer_info.GetName().strip()
        if name:
            return name
    if atom.HasProp("atom_name"):
        name = atom.GetProp("atom_name").strip()
        if name:
            return name
    symbol = atom.GetSymbol()
    atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
    return f"{symbol}{atom_counts[symbol]}"


def read_sdf_structure(path):
    if Chem is None:
        raise ImportError("Reading SDF caps requires RDKit")

    # Preserve the explicit SDF bond table. Full RDKit sanitization can
    # re-aromatize kekulized heterocycles, which loses the input single/double
    # pattern when we later write PDB CONECT records.
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"No readable molecule found in {path}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"No 3D conformer found in {path}")

    conf = mol.GetConformer()
    atom_array = struc.AtomArray(mol.GetNumAtoms())
    coords = []
    atom_names = []
    elements = []
    chain_ids = []
    res_ids = []
    res_names = []
    atom_counts = {}
    metadata_by_index = sdf_atom_metadata(path, mol)

    for atom in mol.GetAtoms():
        atom_metadata = metadata_by_index.get(atom.GetIdx(), {})
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        atom_names.append(rdkit_atom_name(atom, atom_counts, atom_metadata))
        elements.append(atom.GetSymbol())
        chain_ids.append(atom_metadata.get("chain_id") or "A")
        res_names.append(atom_metadata.get("residue_name") or "CAP")
        try:
            res_ids.append(int(atom_metadata.get("residue_number") or 1))
        except ValueError:
            res_ids.append(1)

    atom_array.coord = np.asarray(coords, dtype=float)
    atom_array.chain_id = np.asarray(chain_ids, dtype=object)
    atom_array.res_id = np.asarray(res_ids, dtype=int)
    atom_array.res_name = np.asarray(res_names, dtype=object)
    atom_array.atom_name = np.asarray(atom_names, dtype=object)
    atom_array.element = np.asarray(elements, dtype=object)
    atom_array.hetero = np.ones(mol.GetNumAtoms(), dtype=bool)

    bonds = struc.BondList(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        bonds.add_bond(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            bond_type=bond_order_from_rdkit_bond(bond),
        )
    atom_array.bonds = bonds
    return atom_array


def residue_original_key(row):
    return (
        row.get("label_asym_id", ""),
        row.get("label_seq_id", ""),
        normalize_ins_code(row.get("pdbx_PDB_ins_code", "")),
        row.get("label_comp_id", ""),
        row.get("auth_asym_id", ""),
        row.get("auth_seq_id", ""),
        row.get("auth_comp_id", ""),
    )


def write_atom_site_pdb(
    atom_rows,
    out_path,
    chain_id,
    allowed_receptor_keys=None,
    include_conect=False,
    chem_bond_rows=None,
    struct_conn_rows=None,
):
    atom_serial = 1
    residue_serial = 0
    previous_residue_key = None
    atom_records = []
    atom_lookup = {}
    residue_comp_to_serials = {}
    seen_atoms = set()

    for row in atom_rows:
        element = str(row.get("type_symbol", "")).strip().upper()
        if element in {"H", "D"}:
            continue
        if element and element not in HEAVY_ELEMENTS:
            continue

        if allowed_receptor_keys is not None and not (residue_keys_from_row(row) & allowed_receptor_keys):
            continue

        alt_id = row.get("label_alt_id", ".")
        if alt_id not in (".", "?", "", "A"):
            continue

        res_key = residue_original_key(row)
        if res_key != previous_residue_key:
            residue_serial += 1
            previous_residue_key = res_key

        atom_name = row.get("auth_atom_id") or row.get("label_atom_id")
        res_name = row.get("auth_comp_id") or row.get("label_comp_id") or "UNK"
        record = "HETATM" if str(row.get("group_PDB", "")).upper().startswith("HET") else "ATOM"

        try:
            x = float(row["Cartn_x"])
            y = float(row["Cartn_y"])
            z = float(row["Cartn_z"])
        except Exception:
            continue

        atom_key = (residue_serial, atom_name)
        if atom_key in seen_atoms:
            continue
        seen_atoms.add(atom_key)

        atom_records.append(
            pdb_atom_line(
                atom_serial,
                atom_name,
                res_name,
                chain_id,
                residue_serial,
                x,
                y,
                z,
                element.title(),
                record=record,
            )
        )

        atom_lookup[(residue_serial, atom_name)] = atom_serial
        atom_lookup[
            (
                row.get("label_asym_id", ""),
                row.get("label_seq_id", ""),
                normalize_ins_code(row.get("pdbx_PDB_ins_code", "")),
                row.get("label_comp_id", ""),
                row.get("label_atom_id", ""),
            )
        ] = atom_serial
        residue_comp_to_serials.setdefault((residue_serial, row.get("label_comp_id", "")), set()).add(atom_serial)

        atom_serial += 1

    bonds = {}
    if include_conect:
        serial_to_atom = {}
        for key, serial in atom_lookup.items():
            if isinstance(key, tuple) and len(key) == 2:
                serial_to_atom.setdefault(key[0], {})[key[1]] = serial

        for row in chem_bond_rows or []:
            comp_id = row.get("comp_id", "")
            atom1 = row.get("atom_id_1", "")
            atom2 = row.get("atom_id_2", "")
            order = bond_order_from_value(row.get("value_order", ""))
            for residue_serial, atom_map in serial_to_atom.items():
                if (residue_serial, comp_id) not in residue_comp_to_serials:
                    continue
                if atom1 in atom_map and atom2 in atom_map:
                    add_bond_order(bonds, atom_map[atom1], atom_map[atom2], order)

        for row in struct_conn_rows or []:
            if row.get("conn_type_id", "").lower() not in {"covale", "covale_base"}:
                continue
            order = bond_order_from_value(row.get("pdbx_value_order", row.get("value_order", "")))
            key1 = (
                row.get("ptnr1_label_asym_id", ""),
                row.get("ptnr1_label_seq_id", ""),
                normalize_ins_code(row.get("pdbx_ptnr1_PDB_ins_code", "")),
                row.get("ptnr1_label_comp_id", ""),
                row.get("ptnr1_label_atom_id", ""),
            )
            key2 = (
                row.get("ptnr2_label_asym_id", ""),
                row.get("ptnr2_label_seq_id", ""),
                normalize_ins_code(row.get("pdbx_ptnr2_PDB_ins_code", "")),
                row.get("ptnr2_label_comp_id", ""),
                row.get("ptnr2_label_atom_id", ""),
            )
            if key1 in atom_lookup and key2 in atom_lookup:
                add_bond_order(bonds, atom_lookup[key1], atom_lookup[key2], order)

    with open(out_path, "w") as handle:
        handle.writelines(atom_records)
        if atom_records:
            handle.write("TER\n")
        write_conect_records(handle, bonds)
        handle.write("END\n")

    return len(atom_records)


def load_cap_numbering(yaml_path, cap_name=None):
    '''
    return a dict of {mode: {"C_ter": [atom names], "N_ter": [atom names]}} for the specified cap name
    '''
    with open(yaml_path) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"{yaml_path} does not contain a cap mapping")

    if cap_name is None:
        if len(data) != 1:
            names = ", ".join(sorted(data))
            raise ValueError(f"--cap-name is required because {yaml_path} contains: {names}")
        cap_name = next(iter(data))

    if cap_name not in data:
        raise ValueError(f"Cap name {cap_name!r} not found in {yaml_path}")

    cap_data = data[cap_name]
    if not isinstance(cap_data, dict):
        raise ValueError(f"{cap_name} must map mode names to C_ter/N_ter atom lists")

    parsed = {}
    for mode, mapping in cap_data.items():
        if not isinstance(mapping, dict) or "C_ter" not in mapping or "N_ter" not in mapping:
            raise ValueError(f"{cap_name}.{mode} must contain C_ter and N_ter")
        parsed[mode] = {
            key: [atom.strip() for atom in str(mapping[key]).split(",") if atom.strip()]
            for key in ("C_ter", "N_ter")
        }
        if len(parsed[mode]["C_ter"]) != 5 or len(parsed[mode]["N_ter"]) != 5:
            raise ValueError(f"{cap_name}.{mode} must have exactly five C_ter and five N_ter atoms")

    return cap_name, parsed


def collect_cap_files(cap_dir):
    files = []
    for path in sorted(Path(cap_dir).iterdir()):
        if path.is_file() and path.name.lower().endswith((".sdf", ".sd")):
            files.append(path)
    if not files:
        raise ValueError(f"No .sdf/.sd cap files found in {cap_dir}")
    return files


def build_cap_index(cap_dir, cap_numbering):
    cap_entries = []
    entries_by_mode = {mode: [] for mode in cap_numbering}
    descriptors_by_mode = {mode: [] for mode in cap_numbering}

    for cap_file in collect_cap_files(cap_dir):
        cap_struct = read_sdf_structure(cap_file)
        cap_struct = cap_struct[cap_struct.element != "H"]
        for mode, atom_map in cap_numbering.items():
            coords = np.array(get_atoms_coords(cap_struct, atom_map["C_ter"] + atom_map["N_ter"]))
            entry = {"file": cap_file, "mode": mode}
            descriptors_by_mode[mode].append(cal_descriptor(coords))
            entries_by_mode[mode].append(entry)
            cap_entries.append(entry)

    if not cap_entries:
        raise ValueError("No cap descriptors were generated")

    cap_indexes = {}
    for mode, descriptors in descriptors_by_mode.items():
        if not descriptors:
            continue
        descriptors = np.asarray(descriptors, dtype=np.float32)
        cap_indexes[mode] = {
            "entries": entries_by_mode[mode],
            "tree": BallTree(descriptors, metric="euclidean"),
        }

    return cap_entries, cap_indexes


def ordered_residues(chain_l):
    residues = []
    for res_id in np.unique(chain_l.res_id):
        residues.append(chain_l[chain_l.res_id == res_id])
    return residues


def find_unique_residue(residues, res_name):
    matches = [res for res in residues if str(res.res_name[0]) == res_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {res_name} residue, found {len(matches)}")
    return matches[0]


def residue_after(residues, residue, direction):
    res_id = residue.res_id[0]
    ids = [res.res_id[0] for res in residues]
    idx = ids.index(res_id)
    next_idx = idx + direction
    if next_idx < 0 or next_idx >= len(residues):
        raise ValueError(f"Residue {residue.res_name[0]} has no neighbor in direction {direction}")
    return residues[next_idx]


def residue_atom_target(residue, atom_name):
    return (int(residue.res_id[0]), atom_name)


def terminal_atoms_for_mapping(cap_key, base_side, ace, first_res, nme, last_res):
    if cap_key == "C_ter" and base_side == "last":
        atom_specs = [
            (nme, "CA"),
            (nme, "N"),
            (last_res, "C"),
            (last_res, "O"),
            (last_res, "CA"),
        ]
    elif cap_key == "C_ter" and base_side == "first":
        atom_specs = [
            (first_res, "CA"),
            (first_res, "N"),
            (ace, "C"),
            (ace, "O"),
            (ace, "CA"),
        ]
    elif cap_key == "N_ter" and base_side == "first":
        atom_specs = [
            (ace, "CA"),
            (ace, "C"),
            (ace, "O"),
            (first_res, "N"),
            (first_res, "CA"),
        ]
    elif cap_key == "N_ter" and base_side == "last":
        atom_specs = [
            (last_res, "CA"),
            (last_res, "C"),
            (last_res, "O"),
            (nme, "N"),
            (nme, "CA"),
        ]
    else:
        raise ValueError(f"Cannot map {cap_key} to base side {base_side}")

    coords = []
    targets = []
    for residue, atom_name in atom_specs:
        coords.extend(get_atoms_coords(residue, [atom_name]))
        targets.append(residue_atom_target(residue, atom_name))
    return coords, targets


def build_terminal_definition(chain_l, atom_map):
    residues = ordered_residues(chain_l)
    ace = find_unique_residue(residues, "ACE")
    nme = find_unique_residue(residues, "NME")
    first_res = residue_after(residues, ace, 1)
    last_res = residue_after(residues, nme, -1)

    c_side = "first"
    n_side = "last"

    c_coords, c_targets = terminal_atoms_for_mapping("C_ter", c_side, ace, first_res, nme, last_res)
    n_coords, n_targets = terminal_atoms_for_mapping("N_ter", n_side, ace, first_res, nme, last_res)

    cap_remove_atoms = []
    new_bonds = []

    for key, side_type in (("C_ter", c_side), ("N_ter", n_side)):
        atoms = atom_map[key]
        if key == "C_ter" and side_type == "first":
            cap_remove_atoms.extend(atoms[:2])
            new_bonds.append((int(first_res.res_id[0]), "N", atoms[2]))
        elif key == "C_ter" and side_type == "last":
            cap_remove_atoms.extend(atoms[-3:])
            new_bonds.append((int(last_res.res_id[0]), "C", atoms[1]))
        elif key == "N_ter" and side_type == "first":
            cap_remove_atoms.extend(atoms[-2:])
            new_bonds.append((int(first_res.res_id[0]), "N", atoms[1]))
        elif key == "N_ter" and side_type == "last":
            cap_remove_atoms.extend(atoms[:3])
            new_bonds.append((int(last_res.res_id[0]), "C", atoms[3]))
        else:
            raise ValueError(f"Cannot attach {key} to base side {side_type}")

    return TerminalDefinition(
        ref_coords=np.stack(c_coords + n_coords),
        ligand_targets=c_targets + n_targets,
        remove_res_ids=[int(ace.res_id[0]), int(nme.res_id[0])],
        cap_remove_atoms=cap_remove_atoms,
        new_bonds=new_bonds,
        cap_res_id=int(ace.res_id[0]),
    )


def cap_link_atom_template_name(ligand_atom_name):
    ligand_atom_name = str(ligand_atom_name).strip().upper()
    if ligand_atom_name == "N":
        return "C"
    if ligand_atom_name == "C":
        return "N"
    return None


def place_cap_on_ligand(cap_file, chain_l, atom_map, terminal_def):
    cap_struct = read_sdf_structure(cap_file)
    cap_struct.res_name = ["CAP" for _ in cap_struct.res_name]
    cap_struct = cap_struct[cap_struct.element != "H"]

    cap_coords = np.stack(get_atoms_coords(cap_struct, atom_map["C_ter"] + atom_map["N_ter"]))
    fitted, trans = superimpose(terminal_def.ref_coords, cap_coords)
    rmsd = struc.rmsd(terminal_def.ref_coords, fitted)
    cap_struct = trans.apply(cap_struct)

    averaged = (fitted + terminal_def.ref_coords) / 2
    for atom_name, coord in zip(atom_map["C_ter"] + atom_map["N_ter"], averaged):
        cap_struct[cap_struct.atom_name == atom_name].coord = np.expand_dims(coord, axis=0)

    chain_l = chain_l.copy()
    for (res_id, atom_name), coord in zip(terminal_def.ligand_targets, averaged):
        mask = (chain_l.res_id == res_id) & (chain_l.atom_name == atom_name)
        if np.sum(mask) == 1:
            chain_l.coord[mask] = coord

    keep_ligand = ~np.isin(chain_l.res_id, terminal_def.remove_res_ids)
    chain_l_mod = chain_l[keep_ligand]

    cap_struct.res_id = np.full(cap_struct.array_length(), terminal_def.cap_res_id)
    cap_struct.chain_id = np.full(cap_struct.array_length(), "L")

    cap_keep = np.ones(cap_struct.array_length(), dtype=bool)
    for atom_name in terminal_def.cap_remove_atoms:
        cap_keep &= cap_struct.atom_name != atom_name
    cap_struct_mod = cap_struct[cap_keep]

    merged = struc.concatenate([chain_l_mod, cap_struct_mod])
    bond_list = merged.bonds
    if bond_list is None:
        raise ValueError("Merged ligand does not contain a bond list")

    for ligand_res_id, ligand_atom_name, cap_atom_name in terminal_def.new_bonds:
        ligand_idx = np.where((merged.res_id == ligand_res_id) & (merged.atom_name == ligand_atom_name))[0]
        cap_idx = np.where((merged.res_name == "CAP") & (merged.atom_name == cap_atom_name))[0]
        if len(ligand_idx) != 1 or len(cap_idx) != 1:
            raise ValueError(
                f"Cannot add bond {ligand_res_id}:{ligand_atom_name} - CAP:{cap_atom_name}"
            )
        bond_list.add_bond(int(ligand_idx[0]), int(cap_idx[0]), bond_type=1)
        template_name = cap_link_atom_template_name(ligand_atom_name)
        if template_name:
            merged.atom_name[int(cap_idx[0])] = template_name

    return rmsd, merged


def find_receptor_and_ligand_blocks(blocks):
    receptor_name = None
    ligand_blocks = {}

    for name in blocks:
        if name.endswith("_receptor") and receptor_name is None:
            receptor_name = name
        match = re.search(r"_ligand(\d+)$", name)
        if match:
            ligand_blocks[match.group(1)] = name

    if receptor_name is None:
        receptor_name = next(iter(blocks))

    if not ligand_blocks:
        for name in blocks:
            match = re.search(r"ligand(\d+)", name)
            if match:
                ligand_blocks[match.group(1)] = name

    if not ligand_blocks:
        raise ValueError("No ligand data blocks found")

    return receptor_name, ligand_blocks


def prefilter_candidates(mode_index, query_descriptor, distance_cutoff):
    entries = mode_index["entries"]
    tree = mode_index["tree"]
    query = np.expand_dims(query_descriptor, axis=0)

    indices, distances = tree.query_radius(
        query,
        r=distance_cutoff,
        return_distance=True,
    )
    pairs = sorted(zip(distances[0], indices[0]), key=lambda item: item[0])

    candidates = []
    for distance, index in pairs:
        distance = float(distance)
        candidates.append((distance, entries[int(index)]))

    return candidates


def save_complex_pdb(out_path, capped_ligand, receptor):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serial_by_ligand_index = {}
    serial = 1
    with open(out_path, "w") as handle:
        for i in range(capped_ligand.array_length()):
            atom_name = str(capped_ligand.atom_name[i]).strip()
            res_name = str(capped_ligand.res_name[i]).strip()
            chain_id = str(capped_ligand.chain_id[i]).strip() or "L"
            res_id = int(capped_ligand.res_id[i])
            element = str(capped_ligand.element[i]).strip()
            x, y, z = capped_ligand.coord[i]
            record = "HETATM" if res_name == "CAP" else "ATOM"
            handle.write(
                pdb_atom_line(serial, atom_name, res_name, chain_id, res_id, x, y, z, element, record=record)
            )
            serial_by_ligand_index[i] = serial
            serial += 1

        if capped_ligand.array_length() > 0:
            handle.write("TER\n")

        receptor_start_serial = serial
        for i in range(receptor.array_length()):
            atom_name = str(receptor.atom_name[i]).strip()
            res_name = str(receptor.res_name[i]).strip()
            chain_id = str(receptor.chain_id[i]).strip() or "R"
            res_id = int(receptor.res_id[i])
            element = str(receptor.element[i]).strip()
            x, y, z = receptor.coord[i]
            handle.write(
                pdb_atom_line(serial, atom_name, res_name, chain_id, res_id, x, y, z, element, record="ATOM")
            )
            serial += 1

        if serial > receptor_start_serial:
            handle.write("TER\n")

        bonds = getattr(capped_ligand, "bonds", None)
        if bonds is not None:
            bond_orders = {}
            for bond in bonds._bonds:
                i, j = int(bond[0]), int(bond[1])
                if i not in serial_by_ligand_index or j not in serial_by_ligand_index:
                    continue
                serial1 = serial_by_ligand_index[i]
                serial2 = serial_by_ligand_index[j]
                order = bond_order_from_biotite_row(bond)
                add_bond_order(bond_orders, serial1, serial2, order)
            write_conect_records(handle, bond_orders)

        handle.write("END\n")


def process_base_file(base_file, args, cap_indexes, cap_numbering, tmp_root):
    text = read_text(base_file)
    blocks = split_cif_blocks(text)
    if not blocks:
        raise ValueError("No data blocks found")

    receptor_name, ligand_blocks = find_receptor_and_ligand_blocks(blocks)
    receptor_block = blocks[receptor_name]
    receptor_map = parse_receptor_ligand_map(receptor_block)
    receptor_atoms = parse_loop(receptor_block, "_atom_site.")
    if not receptor_atoms:
        raise ValueError("Receptor block has no _atom_site loop")

    base_stem = strip_structure_suffix(base_file)
    distance_cutoff = args.distance_cutoff
    rows = []

    for ligand_id, ligand_block_name in sorted(ligand_blocks.items(), key=lambda item: int(item[0])):
        if ligand_id not in receptor_map:
            continue

        ligand_block = blocks[ligand_block_name]
        ligand_atoms = parse_loop(ligand_block, "_atom_site.")
        if not ligand_atoms:
            continue

        ligand_tmp = tmp_root / f"{base_stem}_lig{ligand_id}_L.pdb"
        receptor_tmp = tmp_root / f"{base_stem}_lig{ligand_id}_R.pdb"

        try:
            n_ligand_atoms = write_atom_site_pdb(
                ligand_atoms,
                ligand_tmp,
                chain_id="L",
                include_conect=True,
                chem_bond_rows=parse_loop(ligand_block, "_chem_comp_bond."),
                struct_conn_rows=parse_loop(ligand_block, "_struct_conn."),
            )
            n_receptor_atoms = write_atom_site_pdb(
                receptor_atoms,
                receptor_tmp,
                chain_id="R",
                allowed_receptor_keys=receptor_map[ligand_id],
                include_conect=False,
            )

            if n_ligand_atoms == 0 or n_receptor_atoms == 0:
                continue

            chain_l = read_structure(str(ligand_tmp))
            chain_l.chain_id = np.full(chain_l.array_length(), "L")
            receptor = read_structure(str(receptor_tmp))
            receptor.chain_id = np.full(receptor.array_length(), "R")
        finally:
            if not args.keep_tmp:
                for tmp_path in (ligand_tmp, receptor_tmp):
                    try:
                        tmp_path.unlink()
                    except FileNotFoundError:
                        pass

        ligand_hits = []
        ligand_failures = []

        for mode, atom_map in cap_numbering.items():
            mode_index = cap_indexes.get(mode)
            if mode_index is None:
                ligand_failures.append(f"{mode}:prefilter:no_index")
                continue

            try:
                terminal_def = build_terminal_definition(chain_l, atom_map)
            except Exception as exc:
                ligand_failures.append(f"{mode}:terminal:{exc}")
                continue

            query = cal_descriptor(terminal_def.ref_coords)
            candidates = prefilter_candidates(
                mode_index,
                query,
                distance_cutoff=distance_cutoff,
            )
            if not candidates:
                ligand_failures.append(f"{mode}:prefilter:0")
                continue

            for descriptor_distance, cap_entry in candidates:
                try:
                    rmsd, capped_ligand = place_cap_on_ligand(
                        cap_entry["file"],
                        chain_l,
                        atom_map,
                        terminal_def,
                    )
                except Exception as exc:
                    ligand_failures.append(f"{mode}:place:{cap_entry['file'].name}:{exc}")
                    continue

                if rmsd > args.rmsd_cutoff:
                    continue

                try:
                    clash_idx = clash_check(capped_ligand, receptor, thres=args.clash_threshold)
                except Exception as exc:
                    ligand_failures.append(f"{mode}:clash:{cap_entry['file'].name}:{exc}")
                    continue

                if clash_idx:
                    continue

                ligand_hits.append((rmsd, descriptor_distance, cap_entry, capped_ligand))
                if not args.save_all:
                    break

            if ligand_hits and not args.save_all:
                break

        ligand_hits.sort(key=lambda item: item[0])

        hits_to_save = ligand_hits if args.save_all else ligand_hits[:1]
        for rmsd, descriptor_distance, cap_entry, capped_ligand in hits_to_save:
            merged_number, out_path = next_merged_output_path(args.output, args.cyclic_type_name)
            save_complex_pdb(out_path, capped_ligand, receptor)
            row = {
                "cyclic_type_name": args.cyclic_type_name,
                "merged_number": str(merged_number),
                "base_file": str(base_file),
                "base_name": base_stem,
                "ligand_id": ligand_id,
                "cap_file": str(cap_entry["file"]),
                "cap_name": strip_structure_suffix(cap_entry["file"]),
                "mode": cap_entry["mode"],
                "descriptor_distance": f"{descriptor_distance:.6f}",
                "rmsd": f"{rmsd:.6f}",
                "output_file": str(out_path),
                "status": "saved",
                "notes": "",
            }
            append_metadata_row(row)
            rows.append(row)
            if not args.save_all:
                break

        if not ligand_hits:
            rows.append(
                {
                    "cyclic_type_name": "",
                    "merged_number": "",
                    "base_file": str(base_file),
                    "base_name": base_stem,
                    "ligand_id": ligand_id,
                    "cap_file": "",
                    "cap_name": "",
                    "mode": "",
                    "descriptor_distance": "",
                    "rmsd": "",
                    "output_file": "",
                    "status": "no_match",
                    "notes": "; ".join(ligand_failures[:5]),
                }
            )

    return rows


def collect_base_files(base_dir):
    files = []
    for path in sorted(Path(base_dir).iterdir()):
        if path.is_file() and path.name.lower().endswith((".cif.gz", ".cif")):
            files.append(path)
    if not files:
        raise ValueError(f"No .cif/.cif.gz files found in {base_dir}")
    return files


def make_tmp_root(args, cyclic_type_name):
    if args.tmp_dir:
        tmp_parent = Path(args.tmp_dir)
    else:
        tmp_parent = Path.cwd() / f"tmp_for_{sanitize_filename_component(cyclic_type_name)}"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="capping_align_batch_", dir=tmp_parent))


def init_worker(
    args,
    cap_indexes,
    cap_numbering,
    tmp_root,
    merged_counter,
    merged_counter_lock,
    metadata_path,
    metadata_lock,
):
    global WORKER_ARGS
    global WORKER_CAP_INDEXES
    global WORKER_CAP_NUMBERING
    global WORKER_TMP_ROOT
    global WORKER_MERGED_COUNTER
    global WORKER_MERGED_COUNTER_LOCK
    global WORKER_METADATA_PATH
    global WORKER_METADATA_LOCK

    WORKER_ARGS = args
    WORKER_CAP_INDEXES = cap_indexes
    WORKER_CAP_NUMBERING = cap_numbering
    WORKER_TMP_ROOT = Path(tmp_root)
    WORKER_MERGED_COUNTER = merged_counter
    WORKER_MERGED_COUNTER_LOCK = merged_counter_lock
    WORKER_METADATA_PATH = Path(metadata_path)
    WORKER_METADATA_LOCK = metadata_lock


def error_summary_row(base_file, exc):
    return {
        "cyclic_type_name": "",
        "merged_number": "",
        "base_file": str(base_file),
        "base_name": strip_structure_suffix(base_file),
        "ligand_id": "",
        "cap_file": "",
        "cap_name": "",
        "mode": "",
        "descriptor_distance": "",
        "rmsd": "",
        "output_file": "",
        "status": "error",
        "notes": str(exc),
    }


def process_base_file_worker(base_file):
    worker_tmp = WORKER_TMP_ROOT / f"pid_{os.getpid()}"
    worker_tmp.mkdir(parents=True, exist_ok=True)

    try:
        rows = process_base_file(
            Path(base_file),
            WORKER_ARGS,
            WORKER_CAP_INDEXES,
            WORKER_CAP_NUMBERING,
            worker_tmp,
        )
        return {
            "base_file": str(base_file),
            "rows": rows,
            "saved": sum(1 for row in rows if row["status"] == "saved"),
            "error": "",
        }
    except Exception as exc:
        return {
            "base_file": str(base_file),
            "rows": [error_summary_row(base_file, exc)],
            "saved": 0,
            "error": str(exc),
        }


def run_base_files(args, base_files, cap_indexes, cap_numbering, tmp_root, metadata_path):
    processors = args.processors if args.processors > 0 else cpu_count()
    processors = max(1, min(processors, len(base_files)))

    all_rows = []
    initial_number = existing_merged_max_number(args.output, args.cyclic_type_name)

    with Manager() as manager:
        merged_counter = manager.Value("i", initial_number)
        merged_counter_lock = manager.Lock()
        metadata_lock = manager.Lock()
        worker_args = (
            args,
            cap_indexes,
            cap_numbering,
            tmp_root,
            merged_counter,
            merged_counter_lock,
            metadata_path,
            metadata_lock,
        )

        if processors == 1:
            init_worker(*worker_args)
            iterator = tqdm(base_files, total=len(base_files), desc="Base files", unit="file")
            for base_file in iterator:
                result = process_base_file_worker(base_file)
                all_rows.extend(result["rows"])
            return all_rows

        with Pool(
            processes=processors,
            initializer=init_worker,
            initargs=worker_args,
        ) as pool:
            iterator = pool.imap_unordered(process_base_file_worker, base_files, chunksize=1)
            for result in tqdm(iterator, total=len(base_files), desc="Base files", unit="file"):
                all_rows.extend(result["rows"])

    return all_rows


def fill_summary_cyclic_type(rows, cyclic_type_name):
    for row in rows:
        if not row.get("cyclic_type_name"):
            row["cyclic_type_name"] = cyclic_type_name


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align relaxed SDF cap conformers to ligands in CPSea multi-block CIF files, "
            "extract each ligand's mapped receptor residues, reject clashes, and "
            "save capped ligand-receptor PDB complexes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "YAML atom lists describe the cap itself:\n"
            "  C_ter: cap C-terminal amide atoms, usually NME CA,N + cap last-residue C,O,CA\n"
            "  N_ter: cap N-terminal amide atoms, usually ACE CA,C,O + cap first-residue N,CA\n\n"
            "Default mapping:\n"
            "  cap C_ter -> base firstAA CA,N + base ACE C,O,CA\n"
            "  cap N_ter -> base lastAA CA,C,O + base NME N,CA"
        ),
    )
    parser.add_argument("--base", required=True, help="Directory of .cif/.cif.gz base files.")
    parser.add_argument("--cap", required=True, help="Directory of relaxed cap .sdf/.sd files.")
    parser.add_argument("--index", required=True, help="YAML atom mapping for cap C_ter/N_ter atoms.")
    parser.add_argument("-o", "--output", required=True, help="Output directory for capped PDB complexes.")
    parser.add_argument("--cap-name", default=None, help="Top-level cap name in the YAML. Inferred if only one exists.")
    parser.add_argument(
        "--cyclic-type-name",
        default=None,
        help="Prefix for merged structure files. Default: resolved --cap-name.",
    )
    parser.add_argument("--rmsd-cutoff", type=float, default=0.5, help="Maximum anchor RMSD to accept. Default: 0.5")
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=3.0,
        help=(
            "Required hard threshold for Euclidean distance in the 9D terminal descriptor before RMSD."
        ),
    )
    parser.add_argument("--clash-threshold", type=float, default=0.8, help="VDW scaling threshold for clash_check(). Default: 0.8")
    parser.add_argument(
        "-p",
        "--processors",
        type=int,
        default=1,
        help="Number of worker processes over base CIF files. Use 0 for all CPUs. Default: 1",
    )
    parser.add_argument("--save-all", action="store_true", help="Save all clash-free hits in the retrieved candidate set.")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary ligand/receptor PDB files.")
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Parent directory for temporary ligand/receptor PDB files. Default: ./tmp_for_<cyclic-type-name>",
    )
    parser.add_argument("--summary", default=None, help="Summary CSV path. Default: <output>/capping_align_batch_summary.csv")
    parser.add_argument("--metadata", default=None, help="Saved-structure metadata CSV path. Default: <output>/capping_align_batch_metadata.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap_name, cap_numbering = load_cap_numbering(args.index, args.cap_name)
    cyclic_type_name = args.cyclic_type_name or cap_name
    args.cyclic_type_name = cyclic_type_name
    cap_entries, cap_indexes = build_cap_index(args.cap, cap_numbering)
    base_files = collect_base_files(args.base)
    summary_path = Path(args.summary) if args.summary else output_dir / "capping_align_batch_summary.csv"
    metadata_path = Path(args.metadata) if args.metadata else output_dir / "capping_align_batch_metadata.csv"
    write_metadata_header(metadata_path)

    tmp_root = make_tmp_root(args, cyclic_type_name)

    print(f"Cap name: {cap_name}")
    print(f"Cyclic type name: {cyclic_type_name}")
    print(f"Indexed cap entries: {len(cap_entries)}")
    print(f"Base files: {len(base_files)}")
    print("Terminal mapping: cap C_ter -> base ACE-firstAA; cap N_ter -> base lastAA-NME")
    print(f"Descriptor distance cutoff: {args.distance_cutoff:.4f}")
    print("Prefilter: BallTree radius query; no nearest-K truncation")
    print("Output type: pdb")
    requested_processors = args.processors if args.processors > 0 else cpu_count()
    active_processors = max(1, min(requested_processors, len(base_files)))
    print(f"Processors: {active_processors}")
    print(f"Output: {output_dir}")
    print(f"Temporary root: {tmp_root}")

    all_rows = []
    try:
        all_rows = run_base_files(args, base_files, cap_indexes, cap_numbering, tmp_root, metadata_path)
        fill_summary_cyclic_type(all_rows, cyclic_type_name)
    finally:
        if args.keep_tmp:
            print(f"Temporary files kept at: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    with open(summary_path, "w", newline="") as handle:
        fieldnames = [
            "cyclic_type_name",
            "merged_number",
            "base_file",
            "base_name",
            "ligand_id",
            "cap_file",
            "cap_name",
            "mode",
            "descriptor_distance",
            "rmsd",
            "output_file",
            "status",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    saved_total = sum(1 for row in all_rows if row["status"] == "saved")
    print(f"Saved total: {saved_total}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
