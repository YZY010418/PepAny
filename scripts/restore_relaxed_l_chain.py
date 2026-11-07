#!/usr/bin/env python3
"""Restore residue and atom identities in a relaxed, flattened peptide chain.

The affected files contain a chemically intact chain L, including CONECT
records, but every atom has been flattened into one ``LIG`` residue.  This
script uses:

1. the PepAny/CPcomposer raw PDB or BoltzGen mmCIF for amino-acid identities;
2. the unrelaxed merged PDB for coordinate matching to the raw peptide; and
3. the relaxed PDB topology for assigning newly added hydrogens.

No atom coordinate is changed.  The restored PDB keeps the original bond
topology, groups atoms by residue, and renumbers atom serials/CONECT references
consistently.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
except ImportError:
    MMCIF2Dict = None


ATOM_RECORDS = {"ATOM", "HETATM"}
HYDROGEN_ELEMENTS = {"H", "D"}
PLACEHOLDER_RESIDUES = {"ACE", "NME"}
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}
STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
HISTIDINE_NAMES = {"HIS", "HID", "HIE", "HIP"}
RESIDUE_NAME_ALIASES = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
}
UNKNOWN_RESIDUE_NAMES = {"", ".", "?", "UNK", "UNL", "XAA"}
BACKBONE_HEAVY_ATOMS = {"N", "CA", "C", "O", "OXT"}
STANDARD_SIDECHAIN_ATOMS = {
    "GLY": frozenset(),
    "ALA": frozenset({"CB"}),
    "VAL": frozenset({"CB", "CG1", "CG2"}),
    "LEU": frozenset({"CB", "CG", "CD1", "CD2"}),
    "ILE": frozenset({"CB", "CG1", "CG2", "CD1"}),
    "SER": frozenset({"CB", "OG"}),
    "THR": frozenset({"CB", "OG1", "CG2"}),
    "CYS": frozenset({"CB", "SG"}),
    "MET": frozenset({"CB", "CG", "SD", "CE"}),
    "PRO": frozenset({"CB", "CG", "CD"}),
    "ASP": frozenset({"CB", "CG", "OD1", "OD2"}),
    "ASN": frozenset({"CB", "CG", "OD1", "ND2"}),
    "GLU": frozenset({"CB", "CG", "CD", "OE1", "OE2"}),
    "GLN": frozenset({"CB", "CG", "CD", "OE1", "NE2"}),
    "LYS": frozenset({"CB", "CG", "CD", "CE", "NZ"}),
    "ARG": frozenset({"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "HIS": frozenset({"CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "PHE": frozenset({"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "TYR": frozenset(
        {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}
    ),
    "TRP": frozenset(
        {
            "CB",
            "CG",
            "CD1",
            "CD2",
            "NE1",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
        }
    ),
}
BOLTZGEN_CORE_MODES = {
    "disulfidea": "peptide_endpoints",
    "alkenea": "bridge_terminal",
    "amidea": "peptide_endpoints",
    "estera": "peptide_endpoints",
    "thioethera": "n_cap_terminal",
    "thioetherb": "peptide_endpoints",
    "thioetherc": "peptide_endpoints",
    "triazolea": "bridge_terminal",
}


class RestorationError(RuntimeError):
    """Raised when an input cannot be restored safely."""


class TextProgress:
    """Minimal progress fallback for environments without tqdm."""

    def __init__(self, total: int | None, description: str, unit: str) -> None:
        self.total = total
        self.description = description
        self.unit = unit
        self.count = 0
        self.postfix = ""
        self.last_report = time.monotonic()
        self.closed = False
        self._print()

    def _print(self) -> None:
        total = str(self.total) if self.total is not None else "?"
        suffix = f" {self.postfix}" if self.postfix else ""
        print(
            f"{self.description}: {self.count}/{total} {self.unit}{suffix}",
            file=sys.stderr,
            flush=True,
        )

    def update(self, amount: int = 1) -> None:
        self.count += amount
        now = time.monotonic()
        reached_total = self.total is not None and self.count >= self.total
        if now - self.last_report >= 5.0 or reached_total:
            self._print()
            self.last_report = now

    def set_postfix(self, values: dict[str, int], refresh: bool = False) -> None:
        self.postfix = " ".join(f"{key}={value}" for key, value in values.items())
        if refresh:
            self._print()

    def close(self) -> None:
        if self.closed:
            return
        if self.total is None or self.count < self.total:
            self._print()
        self.closed = True


def progress_bar(
    total: int | None,
    description: str,
    unit: str,
    leave: bool = True,
):
    if tqdm is not None:
        return tqdm(
            total=total,
            desc=description,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
        )
    return TextProgress(total, description, unit)


def track_progress(
    iterable: Iterable,
    total: int | None,
    description: str,
    unit: str,
    leave: bool = True,
):
    progress = progress_bar(total, description, unit, leave)
    try:
        for item in iterable:
            yield item
            progress.update()
    finally:
        progress.close()


def progress_write(message: str, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    if tqdm is not None:
        tqdm.write(message, file=stream)
    else:
        print(message, file=stream, flush=True)


@dataclass(frozen=True)
class PDBAtom:
    line_index: int
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str
    line: str

    @property
    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in HYDROGEN_ELEMENTS


@dataclass
class Identity:
    atom_name: str
    resname: str
    resseq: int
    icode: str
    record: str
    element: str

    @property
    def residue_key(self) -> tuple[int, str]:
        return (self.resseq, self.icode)


@dataclass(frozen=True)
class MetadataEntry:
    path: Path
    data: dict
    sample_index: int
    cyclization_type: str
    target_name: str


@dataclass
class RawResidue:
    chain: str
    resname: str
    resseq: int
    icode: str
    atoms: list[PDBAtom]


@dataclass
class RestoredResult:
    relaxed: Path
    metadata: Path
    raw: Path
    merged: Path
    output: Path
    accepted_output: Path | None
    peptide_heavy_atoms: int
    cap_heavy_atoms: int
    hydrogen_atoms: int
    residues: list[str]
    raw_merged_rmsd: float
    raw_merged_max_distance: float
    crosslink_count: int
    added_conect_bonds: int
    accepted_added_conect_bonds: int
    coordinate_check: str
    accepted_coordinate_check: str
    raw_format: str

    def as_dict(self) -> dict:
        return {
            "status": "success",
            "relaxed": str(self.relaxed),
            "metadata": str(self.metadata),
            "raw": str(self.raw),
            "merged": str(self.merged),
            "output": str(self.output),
            "accepted_output": (
                str(self.accepted_output) if self.accepted_output else None
            ),
            "peptide_heavy_atoms": self.peptide_heavy_atoms,
            "cap_heavy_atoms": self.cap_heavy_atoms,
            "hydrogen_atoms": self.hydrogen_atoms,
            "residues": self.residues,
            "raw_merged_rmsd_angstrom": round(self.raw_merged_rmsd, 6),
            "raw_merged_max_distance_angstrom": round(
                self.raw_merged_max_distance, 6
            ),
            "cap_peptide_crosslink_count": self.crosslink_count,
            "added_conect_bonds": self.added_conect_bonds,
            "accepted_added_conect_bonds": self.accepted_added_conect_bonds,
            "coordinate_check": self.coordinate_check,
            "accepted_coordinate_check": self.accepted_coordinate_check,
            "raw_format": self.raw_format,
        }


@dataclass(frozen=True)
class RestorationJob:
    position: int
    relaxed_path: Path
    metadata: MetadataEntry
    raw_path: Path
    merged_path: Path
    output_path: Path
    accepted_output_path: Path | None
    chain: str
    tolerance: float
    overwrite: bool


def distance(atom1: PDBAtom, atom2: PDBAtom) -> float:
    return math.sqrt(
        (atom1.x - atom2.x) ** 2
        + (atom1.y - atom2.y) ** 2
        + (atom1.z - atom2.z) ** 2
    )


def infer_element(atom_name: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", atom_name).upper()
    if not letters:
        return ""
    if letters[:2] in {"CL", "BR"}:
        return letters[:2]
    return letters[0]


def parse_pdb_atom(line: str, line_index: int) -> PDBAtom | None:
    record = line[:6].strip().upper()
    if record not in ATOM_RECORDS:
        return None
    if len(line) < 54:
        raise RestorationError(f"Malformed PDB atom line {line_index + 1}: {line!r}")
    try:
        serial = int(line[6:11])
        resseq = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError as exc:
        raise RestorationError(
            f"Cannot parse PDB atom fields on line {line_index + 1}: {line!r}"
        ) from exc
    name = line[12:16].strip()
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    element = element or infer_element(name)
    return PDBAtom(
        line_index=line_index,
        record=record,
        serial=serial,
        name=name,
        resname=line[17:20].strip(),
        chain=line[21:22].strip(),
        resseq=resseq,
        icode=line[26:27].strip(),
        x=x,
        y=y,
        z=z,
        element=element,
        line=line,
    )


def read_pdb(path: Path) -> tuple[list[str], list[PDBAtom]]:
    lines = path.read_text().splitlines(keepends=True)
    atoms = []
    seen_serials = set()
    for line_index, line in enumerate(lines):
        atom = parse_pdb_atom(line, line_index)
        if atom is None:
            continue
        if atom.serial in seen_serials:
            raise RestorationError(f"{path}: duplicate atom serial {atom.serial}")
        atoms.append(atom)
        seen_serials.add(atom.serial)
    if not atoms:
        raise RestorationError(f"{path}: no ATOM/HETATM records found")
    return lines, atoms


def parse_conect_values(line: str) -> list[int]:
    if not line.startswith("CONECT"):
        return []
    values = []
    payload = line[6:].rstrip("\r\n")
    for start in range(0, len(payload), 5):
        field = payload[start : start + 5].strip()
        if not field:
            continue
        try:
            values.append(int(field))
        except ValueError as exc:
            raise RestorationError(f"Malformed CONECT record: {line.rstrip()}") from exc
    return values


def conect_adjacency(lines: Iterable[str]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for line in lines:
        values = parse_conect_values(line)
        if len(values) < 2:
            continue
        source = values[0]
        for target in values[1:]:
            if target == source:
                continue
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def nested_value(data: dict, *paths: Sequence[str]):
    for path in paths:
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            if current not in (None, ""):
                return current
    return None


def sample_number_from_path(path: Path) -> int | None:
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def sample_number(value: object) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[-1]) if numbers else None


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def load_metadata_entries(path: Path) -> list[MetadataEntry]:
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    if not files:
        raise RestorationError(f"No JSON metadata files found under {path}")
    entries = []
    errors = []
    for metadata_path in track_progress(
        files,
        total=len(files),
        description="Loading metadata",
        unit="json",
    ):
        try:
            data = json.loads(metadata_path.read_text())
            sample_value = nested_value(
                data,
                ("sample_index",),
                ("sample_id",),
                ("evaluation", "sample_index"),
                ("evaluation", "sample_id"),
                ("raw_attempt_metadata", "sample_index"),
            )
            sample = sample_number(sample_value)
            if sample is None:
                sample = sample_number_from_path(metadata_path)
            if sample is None:
                raise ValueError(
                    "sample_index/sample_id is absent and filename has no number"
                )
            cyclization = nested_value(
                data,
                ("cyclization_type",),
                ("evaluation", "run_name"),
                ("raw_attempt_metadata", "run_name"),
            )
            target = nested_value(
                data,
                ("target_name",),
                ("evaluation", "target_id"),
                ("raw_attempt_metadata", "target_id"),
            )
            entries.append(
                MetadataEntry(
                    path=metadata_path,
                    data=data,
                    sample_index=sample,
                    cyclization_type=str(cyclization or ""),
                    target_name=str(target or ""),
                )
            )
        except Exception as exc:
            errors.append(f"{metadata_path}: {exc}")
    if not entries:
        details = "\n".join(errors[:10])
        raise RestorationError(f"No usable metadata JSON found under {path}\n{details}")
    if errors:
        print(
            f"warning: skipped {len(errors)} unusable metadata JSON file(s)",
            file=sys.stderr,
        )
    return entries


def metadata_match_score(relaxed_path: Path, entry: MetadataEntry) -> int:
    path_tokens = {
        normalize_token(part)
        for part in relaxed_path.parts
        if normalize_token(part)
    }
    score = 0
    cyclization = normalize_token(entry.cyclization_type)
    target = normalize_token(entry.target_name)
    if cyclization and cyclization in path_tokens:
        score += 100
    if target and target in path_tokens:
        score += 200
    metadata_parent_tokens = {
        normalize_token(part)
        for part in entry.path.parent.parts
        if normalize_token(part)
    }
    score += 10 * len(path_tokens & metadata_parent_tokens)
    return score


def match_metadata(relaxed_path: Path, entries: Sequence[MetadataEntry]) -> MetadataEntry:
    sample = sample_number_from_path(relaxed_path)
    if sample is None:
        raise RestorationError(
            f"{relaxed_path}: cannot infer a sample number from the filename"
        )
    candidates = [entry for entry in entries if entry.sample_index == sample]
    if not candidates:
        raise RestorationError(
            f"{relaxed_path}: no metadata JSON has sample_index/sample_id "
            f"ending in {sample}"
        )
    ranked = sorted(
        ((metadata_match_score(relaxed_path, entry), entry) for entry in candidates),
        key=lambda item: (item[0], str(item[1].path)),
        reverse=True,
    )
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        tied = "\n  ".join(str(item[1].path) for item in ranked if item[0] == ranked[0][0])
        raise RestorationError(
            f"{relaxed_path}: metadata match is ambiguous for sample {sample}:\n  {tied}"
        )
    return ranked[0][1]


def metadata_source_paths(entry: MetadataEntry, kind: str) -> list[Path]:
    if kind == "raw":
        values = [
            nested_value(entry.data, ("source_files", "raw_candidate_pdb")),
            nested_value(entry.data, ("source_files", "raw_boltzgen_cif")),
            nested_value(entry.data, ("evaluation", "candidate_pdb")),
            nested_value(entry.data, ("evaluation", "generated_cif")),
            nested_value(entry.data, ("raw_attempt_metadata", "candidate_pdb")),
        ]
    elif kind == "merged":
        values = [
            nested_value(entry.data, ("complex_pdb",)),
            nested_value(entry.data, ("source_files", "complex_pdb")),
        ]
    else:
        raise ValueError(kind)
    result = []
    for value in values:
        if value:
            candidate = Path(str(value)).expanduser()
            if candidate not in result:
                result.append(candidate)
    return result


def metadata_raw_format(entry: MetadataEntry, raw_path: Path | None = None) -> str:
    if raw_path is not None and is_cif_path(raw_path):
        return "boltzgen_cif"
    if nested_value(
        entry.data,
        ("source_files", "raw_boltzgen_cif"),
        ("evaluation", "generated_cif"),
    ):
        return "boltzgen_cif"
    if nested_value(
        entry.data, ("assembly", "generated_placeholder_residue_count_removed")
    ) is not None:
        return "cpcomposer_pdb"
    return "pepany_pdb"


def is_cif_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".cif") or name.endswith(".cif.gz")


def source_files_under(root: Path, kind: str) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if kind == "raw":
            if not name.endswith((".pdb", ".cif", ".cif.gz")):
                continue
            if name.endswith(("_native.cif", "_native.cif.gz")):
                continue
        elif not name.endswith(".pdb"):
            continue
        yield path


def source_candidate_score(path: Path, entry: MetadataEntry, kind: str) -> int:
    sample = sample_number_from_path(path)
    if sample != entry.sample_index:
        return -10_000
    normalized_parts = {normalize_token(part) for part in path.parts}
    score = 100
    if normalize_token(entry.cyclization_type) in normalized_parts:
        score += 80
    if normalize_token(entry.target_name) in normalized_parts:
        score += 120
    stem = path.stem.lower()
    parts_lower = {part.lower() for part in path.parts}
    if kind == "raw":
        expected_format = metadata_raw_format(entry)
        if expected_format == "boltzgen_cif":
            score += 80 if is_cif_path(path) else -80
        elif path.suffix.lower() == ".pdb":
            score += 20
        if stem.endswith("_raw"):
            score += 60
        if "candidates" in parts_lower:
            score += 40
        if "relaxed" in stem:
            score -= 100
    else:
        if "complexes" in parts_lower:
            score += 50
        if "raw" in stem or "relaxed" in stem:
            score -= 100
    return score


def best_source_in_roots(
    roots: Sequence[Path],
    entry: MetadataEntry,
    kind: str,
    path_cache: dict[tuple[Path, str], list[Path]] | None = None,
) -> Path | None:
    if path_cache is None:
        path_cache = {}
    candidates = []
    seen = set()
    for root in roots:
        if root is None or not root.exists():
            continue
        cache_key = (root.resolve(), kind)
        if cache_key not in path_cache:
            path_cache[cache_key] = list(
                track_progress(
                    source_files_under(root, kind),
                    total=None,
                    description=f"Indexing {root.name}",
                    unit="structure",
                    leave=False,
                )
            )
        paths = path_cache[cache_key]
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            score = source_candidate_score(path, entry, kind)
            if score > -10_000:
                candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        tied = [path for score, path in candidates if score == candidates[0][0]]
        raise RestorationError(
            f"Ambiguous {kind} structure for {entry.path}:\n  "
            + "\n  ".join(str(path) for path in tied[:20])
        )
    return candidates[0][1]


def resolve_source_pdb(
    entry: MetadataEntry,
    kind: str,
    relaxed_path: Path,
    metadata_root: Path,
    explicit_root: Path | None,
    path_cache: dict[tuple[Path, str], list[Path]] | None = None,
) -> Path:
    for candidate in metadata_source_paths(entry, kind):
        if candidate.is_file():
            return candidate

    if explicit_root is not None and explicit_root.is_file():
        if explicit_root.resolve() == relaxed_path.resolve():
            raise RestorationError(
                f"Explicit {kind} structure is the relaxed input itself: "
                f"{explicit_root}"
            )
        return explicit_root

    roots = []
    if explicit_root is not None:
        roots.append(explicit_root)

    if kind == "merged":
        if metadata_root.is_dir():
            try:
                relative = entry.path.relative_to(metadata_root).with_suffix(".pdb")
                roots.append(metadata_root.parent / "complexes" / relative)
            except ValueError:
                pass
        unrelaxed_sibling = relaxed_path.with_name(
            re.sub(r"_relaxed$", "", relaxed_path.stem) + ".pdb"
        )
        if unrelaxed_sibling.is_file() and unrelaxed_sibling.resolve() != relaxed_path.resolve():
            return unrelaxed_sibling
        roots.append(relaxed_path.parent)
    else:
        sample = entry.sample_index
        raw_sibling = relaxed_path.with_name(f"{sample}_raw.pdb")
        if raw_sibling.is_file() and raw_sibling.resolve() != relaxed_path.resolve():
            return raw_sibling
        dataset_parent = (
            metadata_root.parent.parent
            if metadata_root.is_dir() and metadata_root.name == "metadata"
            else metadata_root.parent
        )
        old_raw_paths = metadata_source_paths(entry, "raw")
        raw_dataset_names = {
            part
            for path in old_raw_paths
            for part in path.parts
            if "raw" in part.lower() and "pepany" in part.lower()
        }
        raw_dataset_names.add("pepany_cptest_raw")
        roots.extend(dataset_parent / name for name in sorted(raw_dataset_names))
        roots.append(relaxed_path.parent)

    for root in roots:
        if root.is_file() and root.resolve() != relaxed_path.resolve():
            return root

    sample_id = nested_value(
        entry.data,
        ("sample_id",),
        ("evaluation", "sample_id"),
    )
    sample_stems = [str(entry.sample_index)]
    if sample_id:
        sample_stems.insert(0, str(sample_id))
    suffixes = (".pdb", ".cif", ".cif.gz") if kind == "raw" else (".pdb",)
    sample_filenames = [
        f"{stem}{suffix}" for stem in dict.fromkeys(sample_stems) for suffix in suffixes
    ]
    if kind == "raw":
        relative_parents = [
            Path(entry.cyclization_type) / "candidates" / entry.target_name,
            Path(entry.cyclization_type)
            / entry.target_name
            / "intermediate_designs",
            Path(entry.cyclization_type) / entry.target_name,
            Path("candidates") / entry.target_name,
            Path(entry.target_name) / "intermediate_designs",
            Path(entry.target_name),
        ]
    else:
        relative_parents = [
            Path(entry.cyclization_type) / entry.target_name,
            Path(entry.target_name),
        ]
    relative_candidates = [
        parent / filename
        for parent in relative_parents
        for filename in sample_filenames
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for relative in relative_candidates:
            candidate = root / relative
            if candidate.is_file() and candidate.resolve() != relaxed_path.resolve():
                return candidate

    source = best_source_in_roots(roots, entry, kind, path_cache)
    if source is not None and source.resolve() != relaxed_path.resolve():
        return source

    old_paths = metadata_source_paths(entry, kind)
    old_hint = str(old_paths[0]) if old_paths else "(metadata has no path)"
    option = "--raw-dir" if kind == "raw" else "--merged-dir"
    raise RestorationError(
        f"Cannot locate {kind} structure for {relaxed_path}. "
        f"Metadata path: {old_hint}. "
        f"Provide {option} to relocate the old path."
    )


def pdb_residues(
    atoms: Sequence[PDBAtom], chain: str
) -> list[RawResidue]:
    residues: dict[tuple[int, str, str], RawResidue] = {}
    for atom in atoms:
        if atom.chain != chain or atom.is_hydrogen:
            continue
        key = (atom.resseq, atom.icode, atom.resname)
        residue = residues.setdefault(
            key,
            RawResidue(
                chain=chain,
                resname=atom.resname,
                resseq=atom.resseq,
                icode=atom.icode,
                atoms=[],
            ),
        )
        residue.atoms.append(atom)
    return list(residues.values())


def integer_metadata_value(entry: MetadataEntry, path: Sequence[str]) -> int | None:
    value = nested_value(entry.data, path)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RestorationError(
            f"{entry.path}: metadata {'.'.join(path)} is not an integer: {value!r}"
        ) from exc


def cpcomposer_peptide_atoms(
    residues: Sequence[RawResidue], entry: MetadataEntry
) -> tuple[list[PDBAtom], list[PDBAtom]]:
    removed_residue_count = integer_metadata_value(
        entry, ("assembly", "generated_placeholder_residue_count_removed")
    )
    if removed_residue_count != 2:
        raise RestorationError(
            f"{entry.path}: CPcomposer assembly reports "
            f"generated_placeholder_residue_count_removed={removed_residue_count}; "
            "only the documented first+last two-placeholder scheme is supported"
        )
    if len(residues) < 3:
        raise RestorationError(
            f"{entry.path}: cannot remove first/last placeholders from only "
            f"{len(residues)} raw peptide residue(s)"
        )

    placeholder_residues = [residues[0], residues[-1]]
    core_residues = list(residues[1:-1])
    placeholders = [
        atom for residue in placeholder_residues for atom in residue.atoms
    ]
    peptide = [atom for residue in core_residues for atom in residue.atoms]

    expected_atom_count = integer_metadata_value(
        entry, ("assembly", "generated_placeholder_atom_count_removed")
    )
    if expected_atom_count is not None and expected_atom_count != len(placeholders):
        raise RestorationError(
            f"{entry.path}: CPcomposer placeholder heavy-atom count differs from "
            f"metadata: raw first+last residues contain {len(placeholders)}, "
            f"metadata reports {expected_atom_count}"
        )
    expected_core_residues = integer_metadata_value(
        entry, ("assembly", "linear_core_residue_count")
    )
    if (
        expected_core_residues is not None
        and expected_core_residues != len(core_residues)
    ):
        raise RestorationError(
            f"{entry.path}: CPcomposer core residue count differs from metadata: "
            f"{len(core_residues)} != {expected_core_residues}"
        )
    return peptide, placeholders


def raw_peptide_atoms(
    raw_atoms: Sequence[PDBAtom], entry: MetadataEntry, chain: str
) -> tuple[list[PDBAtom], list[PDBAtom]]:
    source_chain = str(
        nested_value(
            entry.data,
            ("evaluation", "ligand_chain"),
            ("raw_attempt_metadata", "ligand_chain"),
        )
        or chain
    )
    heavy = [
        atom
        for atom in raw_atoms
        if atom.chain == source_chain and not atom.is_hydrogen
    ]
    if not heavy:
        available = sorted({atom.chain or "(blank)" for atom in raw_atoms})
        raise RestorationError(
            f"{entry.path}: raw PDB has no heavy atoms in chain {source_chain!r}; "
            f"available chains: {available}"
        )

    residues = pdb_residues(raw_atoms, source_chain)
    if nested_value(
        entry.data, ("assembly", "generated_placeholder_residue_count_removed")
    ) is not None:
        peptide, placeholders = cpcomposer_peptide_atoms(residues, entry)
        validate_raw_residue_names(peptide)
        return peptide, placeholders

    terminal_indexes = nested_value(
        entry.data, ("merge", "generated_terminal_atom_indexes")
    )
    removed_positions: set[int] = set()
    if isinstance(terminal_indexes, list) and len(terminal_indexes) == 10:
        try:
            terminal_indexes = [int(index) for index in terminal_indexes]
            removed_positions = {
                terminal_indexes[0],
                terminal_indexes[1],
                terminal_indexes[2],
                terminal_indexes[8],
                terminal_indexes[9],
            }
        except (TypeError, ValueError):
            removed_positions = set()
        if removed_positions and not all(0 <= index < len(heavy) for index in removed_positions):
            removed_positions = set()

    if removed_positions:
        peptide = [atom for index, atom in enumerate(heavy) if index not in removed_positions]
        placeholders = [atom for index, atom in enumerate(heavy) if index in removed_positions]
    else:
        peptide = [atom for atom in heavy if atom.resname not in PLACEHOLDER_RESIDUES]
        placeholders = [atom for atom in heavy if atom.resname in PLACEHOLDER_RESIDUES]

    if not peptide:
        raise RestorationError("Raw chain contains no peptide atoms after cap removal")
    validate_raw_residue_names(peptide)
    return peptide, placeholders


def validate_raw_residue_names(peptide: Sequence[PDBAtom]) -> None:
    unknown = sorted({atom.resname for atom in peptide if atom.resname not in STANDARD_RESIDUES})
    if unknown:
        raise RestorationError(
            "Raw peptide contains unsupported/non-standard residue(s): "
            + ", ".join(unknown)
        )


def read_cif_text(path: Path) -> str:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def cif_column(
    dictionary: dict, key: str, length: int, default: str | None = None
) -> list[str]:
    value = dictionary.get(key)
    if value is None:
        return [default] * length
    values = value if isinstance(value, list) else [value]
    if len(values) != length:
        raise RestorationError(
            f"mmCIF column {key} has {len(values)} values, expected {length}"
        )
    return [str(item) for item in values]


def preferred_cif_value(primary: str, fallback: str) -> str:
    return fallback if primary in {".", "?", ""} else primary


def cif_resseq(value: str, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def read_cif_residue_chains(path: Path) -> list[dict]:
    if MMCIF2Dict is None:
        raise RestorationError(
            "BoltzGen CIF input requires Biopython "
            "(install with `conda install biopython`)"
        )
    raw_blocks = [
        block
        for block in re.split(r"(?m)(?=^data_)", read_cif_text(path))
        if block.strip()
    ]
    chains = []
    atom_serial = 1
    for block_index, block_text in enumerate(raw_blocks):
        dictionary = MMCIF2Dict(io.StringIO(block_text))
        x_values = dictionary.get("_atom_site.Cartn_x")
        if x_values is None:
            continue
        x_values = x_values if isinstance(x_values, list) else [x_values]
        x_values = [str(value) for value in x_values]
        atom_count = len(x_values)
        block_name = str(dictionary.get("data_", f"block{block_index + 1}"))
        y_values = cif_column(dictionary, "_atom_site.Cartn_y", atom_count)
        z_values = cif_column(dictionary, "_atom_site.Cartn_z", atom_count)
        label_chains = cif_column(
            dictionary, "_atom_site.label_asym_id", atom_count, "?"
        )
        auth_chains = cif_column(
            dictionary, "_atom_site.auth_asym_id", atom_count, "?"
        )
        label_resnames = cif_column(
            dictionary, "_atom_site.label_comp_id", atom_count, "UNK"
        )
        auth_resnames = cif_column(
            dictionary, "_atom_site.auth_comp_id", atom_count, "?"
        )
        label_seq_ids = cif_column(
            dictionary, "_atom_site.label_seq_id", atom_count, "?"
        )
        auth_seq_ids = cif_column(
            dictionary, "_atom_site.auth_seq_id", atom_count, "?"
        )
        label_atom_names = cif_column(
            dictionary, "_atom_site.label_atom_id", atom_count, "?"
        )
        auth_atom_names = cif_column(
            dictionary, "_atom_site.auth_atom_id", atom_count, "?"
        )
        elements = cif_column(
            dictionary, "_atom_site.type_symbol", atom_count, "?"
        )
        insertion_codes = cif_column(
            dictionary, "_atom_site.pdbx_PDB_ins_code", atom_count, "."
        )
        altlocs = cif_column(
            dictionary, "_atom_site.label_alt_id", atom_count, "."
        )
        model_numbers = cif_column(
            dictionary, "_atom_site.pdbx_PDB_model_num", atom_count, "1"
        )
        first_model = model_numbers[0]

        chains_by_id: dict[str, dict] = {}
        for index in range(atom_count):
            if model_numbers[index] != first_model or altlocs[index] not in {
                ".",
                "?",
                "A",
            }:
                continue
            chain_id = preferred_cif_value(
                auth_chains[index], label_chains[index]
            )
            resname = preferred_cif_value(
                auth_resnames[index], label_resnames[index]
            ).upper()
            seq_id = preferred_cif_value(
                auth_seq_ids[index], label_seq_ids[index]
            )
            atom_name = preferred_cif_value(
                auth_atom_names[index], label_atom_names[index]
            )
            icode = insertion_codes[index]
            icode = "" if icode in {".", "?"} else icode
            chain_record = chains_by_id.setdefault(
                chain_id,
                {
                    "id": chain_id,
                    "block": block_name,
                    "residue_map": {},
                    "residues": [],
                },
            )
            residue_key = (seq_id, icode, resname)
            residue = chain_record["residue_map"].get(residue_key)
            if residue is None:
                residue = RawResidue(
                    chain=chain_id,
                    resname=resname,
                    resseq=cif_resseq(
                        seq_id, len(chain_record["residues"]) + 1
                    ),
                    icode=icode,
                    atoms=[],
                )
                chain_record["residue_map"][residue_key] = residue
                chain_record["residues"].append(residue)
            if any(atom.name == atom_name for atom in residue.atoms):
                continue
            element = elements[index].strip().upper()
            if element in {"", ".", "?"}:
                element = infer_element(atom_name)
            try:
                coordinates = (
                    float(x_values[index]),
                    float(y_values[index]),
                    float(z_values[index]),
                )
            except ValueError as exc:
                raise RestorationError(
                    f"{path}: invalid coordinates for {chain_id}:"
                    f"{resname}{seq_id}:{atom_name}"
                ) from exc
            residue.atoms.append(
                PDBAtom(
                    line_index=atom_serial - 1,
                    record="ATOM" if resname in STANDARD_RESIDUES else "HETATM",
                    serial=atom_serial,
                    name=atom_name,
                    resname=resname,
                    chain=chain_id,
                    resseq=residue.resseq,
                    icode=icode,
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                    element=element,
                    line="",
                )
            )
            atom_serial += 1
        chains.extend(chains_by_id.values())

    if not chains:
        raise RestorationError(f"{path}: no mmCIF _atom_site coordinates found")
    id_counts = Counter(chain["id"] for chain in chains)
    for chain_record in chains:
        chain_record["display_id"] = (
            chain_record["id"]
            if id_counts[chain_record["id"]] == 1
            else f"{chain_record['block']}:{chain_record['id']}"
        )
        chain_record.pop("residue_map", None)
    return chains


def complete_peptide_residues(chain_record: dict) -> list[RawResidue]:
    return [
        residue
        for residue in chain_record["residues"]
        if residue.resname not in {"HOH", "WAT"}
        and {"N", "CA", "C", "O"}.issubset(
            {atom.name for atom in residue.atoms}
        )
    ]


def select_boltzgen_peptide_chain(
    chains: Sequence[dict], requested: str, raw_path: Path
) -> dict:
    if requested and requested.lower() != "auto":
        matches = [
            chain_record
            for chain_record in chains
            if requested
            in {chain_record["id"], chain_record["display_id"]}
        ]
        if len(matches) != 1:
            available = ", ".join(
                str(chain_record["display_id"]) for chain_record in chains
            )
            raise RestorationError(
                f"{raw_path}: BoltzGen peptide chain {requested!r} is not unique; "
                f"available chains: {available}"
            )
        return matches[0]

    candidates = []
    for chain_record in chains:
        residue_count = len(complete_peptide_residues(chain_record))
        if residue_count >= 2:
            candidates.append((residue_count, chain_record))
    if not candidates:
        raise RestorationError(
            f"{raw_path}: cannot detect a BoltzGen chain with at least two "
            "complete peptide residues"
        )
    candidates.sort(key=lambda item: (item[0], str(item[1]["display_id"])))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RestorationError(
            f"{raw_path}: BoltzGen peptide-chain auto-detection is ambiguous; "
            "metadata must contain evaluation.peptide_chain"
        )
    return candidates[0][1]


def boltzgen_peptide_atoms(
    raw_path: Path, entry: MetadataEntry
) -> tuple[list[PDBAtom], list[PDBAtom]]:
    chains = read_cif_residue_chains(raw_path)
    requested_chain = str(
        nested_value(entry.data, ("evaluation", "peptide_chain")) or "auto"
    )
    peptide_chain = select_boltzgen_peptide_chain(
        chains, requested_chain, raw_path
    )
    peptide_residues = complete_peptide_residues(peptide_chain)
    condition_key = normalize_token(entry.cyclization_type)
    mode = BOLTZGEN_CORE_MODES.get(condition_key)
    if mode is None:
        raise RestorationError(
            f"{entry.path}: unsupported BoltzGen cyclization type "
            f"{entry.cyclization_type!r}"
        )

    if mode == "bridge_terminal":
        bridge_name = "2AG" if condition_key == "alkenea" else "HIX"
        peptide_residues = [
            residue
            for residue in peptide_residues
            if residue.resname != bridge_name
        ]
        core_residues = peptide_residues[1:]
    elif mode == "n_cap_terminal":
        core_residues = peptide_residues[:-1]
    else:
        core_residues = peptide_residues[1:-1]
    if not core_residues:
        raise RestorationError(
            f"{raw_path}: no BoltzGen peptide core remains for mode {mode}"
        )

    core_ids = {id(residue) for residue in core_residues}
    placeholders = [
        atom
        for residue in peptide_residues
        if id(residue) not in core_ids
        for atom in residue.atoms
        if not atom.is_hydrogen
    ]
    peptide = [
        atom
        for residue in core_residues
        for atom in residue.atoms
        if not atom.is_hydrogen
    ]
    expected_residue_count = integer_metadata_value(
        entry, ("assembly", "linear_core_residue_count")
    )
    if (
        expected_residue_count is not None
        and expected_residue_count != len(core_residues)
    ):
        raise RestorationError(
            f"{entry.path}: BoltzGen core residue count differs from metadata: "
            f"{len(core_residues)} != {expected_residue_count}"
        )
    expected_atom_count = integer_metadata_value(
        entry, ("assembly", "linear_core_heavy_atom_count")
    )
    if expected_atom_count is not None and expected_atom_count != len(peptide):
        raise RestorationError(
            f"{entry.path}: BoltzGen core heavy-atom count differs from metadata: "
            f"{len(peptide)} != {expected_atom_count}"
        )
    validate_raw_residue_names(peptide)
    return peptide, placeholders


def load_raw_peptide_atoms(
    raw_path: Path, entry: MetadataEntry, chain: str
) -> tuple[list[PDBAtom], list[PDBAtom], str]:
    raw_format = metadata_raw_format(entry, raw_path)
    if raw_format == "boltzgen_cif":
        peptide, placeholders = boltzgen_peptide_atoms(raw_path, entry)
    else:
        _, raw_atoms = read_pdb(raw_path)
        peptide, placeholders = raw_peptide_atoms(raw_atoms, entry, chain)
    return peptide, placeholders, raw_format


def hungarian_assignment(costs: Sequence[Sequence[float]]) -> list[int]:
    """Return the selected column for every row of an n x m matrix (n <= m)."""
    n = len(costs)
    m = len(costs[0]) if n else 0
    if n == 0:
        return []
    if n > m:
        raise RestorationError(f"Cannot assign {n} source atoms to {m} target atoms")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        min_values = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < min_values[j]:
                    min_values[j] = current
                    way[j] = j0
                if min_values[j] < delta:
                    delta = min_values[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_values[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    if any(column < 0 for column in assignment):
        raise RestorationError("Internal error: incomplete coordinate assignment")
    return assignment


def contiguous_mapping(
    raw_atoms: Sequence[PDBAtom],
    merged_atoms: Sequence[PDBAtom],
    tolerance: float,
) -> dict[int, int] | None:
    count = len(raw_atoms)
    candidates = []
    raw_elements = [atom.element for atom in raw_atoms]
    for start in range(0, len(merged_atoms) - count + 1):
        window = merged_atoms[start : start + count]
        if [atom.element for atom in window] != raw_elements:
            continue
        distances = [
            distance(raw_atom, merged_atom)
            for raw_atom, merged_atom in zip(raw_atoms, window)
        ]
        if max(distances, default=0.0) <= tolerance:
            rmsd = math.sqrt(sum(value * value for value in distances) / len(distances))
            candidates.append((rmsd, start))
    if not candidates:
        return None
    _, start = min(candidates)
    return {index: start + index for index in range(count)}


def coordinate_mapping(
    raw_atoms: Sequence[PDBAtom],
    merged_atoms: Sequence[PDBAtom],
    tolerance: float,
) -> dict[int, int]:
    direct = contiguous_mapping(raw_atoms, merged_atoms, tolerance)
    if direct is not None:
        return direct

    mapping = {}
    raw_by_element: dict[str, list[int]] = defaultdict(list)
    merged_by_element: dict[str, list[int]] = defaultdict(list)
    for index, atom in enumerate(raw_atoms):
        raw_by_element[atom.element].append(index)
    for index, atom in enumerate(merged_atoms):
        merged_by_element[atom.element].append(index)

    for element, raw_indexes in raw_by_element.items():
        target_indexes = merged_by_element.get(element, [])
        if len(raw_indexes) > len(target_indexes):
            raise RestorationError(
                f"Merged structure has too few {element} atoms: "
                f"{len(target_indexes)} < {len(raw_indexes)}"
            )
        costs = []
        for raw_rank, raw_index in enumerate(raw_indexes):
            row = []
            for merged_rank, merged_index in enumerate(target_indexes):
                coordinate_cost = distance(raw_atoms[raw_index], merged_atoms[merged_index])
                order_cost = 0.02 * abs(
                    raw_rank / max(1, len(raw_indexes) - 1)
                    - merged_rank / max(1, len(target_indexes) - 1)
                )
                row.append(coordinate_cost + order_cost)
            costs.append(row)
        selected_columns = hungarian_assignment(costs)
        for raw_index, column in zip(raw_indexes, selected_columns):
            mapping[raw_index] = target_indexes[column]

    distances = [
        distance(raw_atoms[raw_index], merged_atoms[merged_index])
        for raw_index, merged_index in mapping.items()
    ]
    if max(distances, default=0.0) > tolerance:
        worst_raw_index = max(
            mapping,
            key=lambda index: distance(raw_atoms[index], merged_atoms[mapping[index]]),
        )
        worst_merged_index = mapping[worst_raw_index]
        raw_atom = raw_atoms[worst_raw_index]
        merged_atom = merged_atoms[worst_merged_index]
        raise RestorationError(
            "Raw→merged coordinate match exceeds tolerance: "
            f"{raw_atom.resname}{raw_atom.resseq}:{raw_atom.name} → "
            f"{merged_atom.name}, {distance(raw_atom, merged_atom):.3f} Å "
            f"> {tolerance:.3f} Å"
        )
    return mapping


def expected_raw_peptide_edges(atoms: Sequence[PDBAtom]) -> set[tuple[int, int]]:
    edges = set()
    for i, atom1 in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            atom2 = atoms[j]
            same_residue = (atom1.resseq, atom1.icode) == (atom2.resseq, atom2.icode)
            consecutive_peptide = (
                atom1.name == "C"
                and atom2.name == "N"
                and atom2.resseq == atom1.resseq + 1
            ) or (
                atom2.name == "C"
                and atom1.name == "N"
                and atom1.resseq == atom2.resseq + 1
            )
            if not same_residue and not consecutive_peptide:
                continue
            radius1 = COVALENT_RADII.get(atom1.element, 0.75)
            radius2 = COVALENT_RADII.get(atom2.element, 0.75)
            cutoff = radius1 + radius2 + 0.30
            if 0.4 < distance(atom1, atom2) <= cutoff:
                edges.add((i, j))
    return edges


def validate_peptide_mapping(
    raw_atoms: Sequence[PDBAtom],
    merged_atoms: Sequence[PDBAtom],
    mapping: dict[int, int],
    merged_adjacency: dict[int, set[int]],
) -> None:
    expected = expected_raw_peptide_edges(raw_atoms)
    if not expected:
        raise RestorationError("Could not infer any peptide bonds from the raw PDB")
    missing = []
    for raw_i, raw_j in expected:
        serial_i = merged_atoms[mapping[raw_i]].serial
        serial_j = merged_atoms[mapping[raw_j]].serial
        if serial_j not in merged_adjacency.get(serial_i, set()):
            missing.append((raw_i, raw_j))
    allowed_missing = max(1, int(0.05 * len(expected)))
    if len(missing) > allowed_missing:
        examples = ", ".join(
            f"{raw_atoms[i].resname}{raw_atoms[i].resseq}:{raw_atoms[i].name}-"
            f"{raw_atoms[j].resname}{raw_atoms[j].resseq}:{raw_atoms[j].name}"
            for i, j in missing[:6]
        )
        raise RestorationError(
            f"Raw→merged mapping fails topology validation: "
            f"{len(missing)}/{len(expected)} expected bonds missing ({examples})"
        )


def merged_to_relaxed_mapping(
    merged_atoms: Sequence[PDBAtom],
    relaxed_atoms: Sequence[PDBAtom],
    merged_adjacency: dict[int, set[int]],
    relaxed_adjacency: dict[int, set[int]],
) -> dict[int, int]:
    if len(merged_atoms) != len(relaxed_atoms):
        raise RestorationError(
            f"Merged/relaxed heavy atom counts differ: "
            f"{len(merged_atoms)} != {len(relaxed_atoms)}"
        )

    if all(
        merged.element == relaxed.element and merged.name == relaxed.name
        for merged, relaxed in zip(merged_atoms, relaxed_atoms)
    ):
        return {index: index for index in range(len(merged_atoms))}

    merged_keys: dict[tuple[str, str], list[int]] = defaultdict(list)
    relaxed_keys: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, atom in enumerate(merged_atoms):
        merged_keys[(atom.element, atom.name)].append(index)
    for index, atom in enumerate(relaxed_atoms):
        relaxed_keys[(atom.element, atom.name)].append(index)
    if merged_keys.keys() == relaxed_keys.keys() and all(
        len(merged_keys[key]) == len(relaxed_keys[key]) == 1 for key in merged_keys
    ):
        return {
            indexes[0]: relaxed_keys[key][0]
            for key, indexes in merged_keys.items()
        }

    if all(
        merged.element == relaxed.element
        for merged, relaxed in zip(merged_atoms, relaxed_atoms)
    ):
        relaxed_heavy_serials = {atom.serial for atom in relaxed_atoms}
        degree_matches = 0
        for merged, relaxed in zip(merged_atoms, relaxed_atoms):
            merged_degree = len(merged_adjacency.get(merged.serial, set()))
            relaxed_heavy_degree = sum(
                1
                for serial in relaxed_adjacency.get(relaxed.serial, set())
                if serial in relaxed_heavy_serials
            )
            degree_matches += merged_degree == relaxed_heavy_degree
        if degree_matches >= int(0.95 * len(merged_atoms)):
            return {index: index for index in range(len(merged_atoms))}

    raise RestorationError(
        "Cannot establish merged→relaxed heavy atom identity. Atom order and "
        "unique generic atom names both changed."
    )


def generic_cap_names(
    cap_indexes: Sequence[int],
    merged_atoms: Sequence[PDBAtom],
    peptide_by_merged_index: dict[int, PDBAtom],
    adjacency: dict[int, set[int]],
) -> tuple[dict[int, str], list[tuple[int, int]]]:
    serial_to_index = {atom.serial: index for index, atom in enumerate(merged_atoms)}
    cap_set = set(cap_indexes)
    forced_names: dict[int, str] = {}
    crosslinks = []
    for peptide_index, peptide_atom in peptide_by_merged_index.items():
        for neighbor_serial in adjacency.get(merged_atoms[peptide_index].serial, set()):
            cap_index = serial_to_index.get(neighbor_serial)
            if cap_index not in cap_set:
                continue
            crosslinks.append((peptide_index, cap_index))
            if peptide_atom.name == "N":
                forced_names[cap_index] = "C"
            elif peptide_atom.name == "C":
                forced_names[cap_index] = "N"

    if len(crosslinks) != 2:
        raise RestorationError(
            f"Expected exactly 2 peptide–CAP CONECT bonds, found {len(crosslinks)}"
        )
    if set(forced_names.values()) != {"C", "N"}:
        details = ", ".join(
            f"{peptide_by_merged_index[p].name}-{merged_atoms[c].name}"
            for p, c in crosslinks
        )
        raise RestorationError(
            "CAP links are not attached to one peptide N and one peptide C: " + details
        )

    reserved = {"C", "N"}
    used = set()
    counts: dict[str, int] = defaultdict(int)
    result = {}
    for cap_index in cap_indexes:
        if cap_index in forced_names:
            name = forced_names[cap_index]
        else:
            element = merged_atoms[cap_index].element.upper()
            counts[element] += 1
            name = f"{element}{counts[element]}"
            while name in reserved or name in used:
                counts[element] += 1
                name = f"{element}{counts[element]}"
        if len(name) > 4:
            raise RestorationError(f"Generated CAP atom name exceeds PDB limit: {name}")
        if name in used:
            raise RestorationError(f"Duplicate generated CAP atom name: {name}")
        used.add(name)
        result[cap_index] = name
    return result, crosslinks


def nearest_heavy_parent(
    hydrogen: PDBAtom, heavy_atoms: Sequence[PDBAtom], max_distance: float = 1.35
) -> PDBAtom | None:
    ranked = sorted((distance(hydrogen, atom), atom) for atom in heavy_atoms)
    if ranked and ranked[0][0] <= max_distance:
        return ranked[0][1]
    return None


def hydrogen_parent_groups(
    relaxed_atoms: Sequence[PDBAtom],
    identities: dict[int, Identity],
    adjacency: dict[int, set[int]],
) -> dict[int, list[PDBAtom]]:
    atom_by_serial = {atom.serial: atom for atom in relaxed_atoms}
    heavy_atoms = [
        atom
        for atom in relaxed_atoms
        if atom.serial in identities and not atom.is_hydrogen
    ]
    heavy_serials = {atom.serial for atom in heavy_atoms}
    groups: dict[int, list[PDBAtom]] = defaultdict(list)
    for hydrogen in (atom for atom in relaxed_atoms if atom.is_hydrogen):
        parents = [
            serial
            for serial in adjacency.get(hydrogen.serial, set())
            if serial in heavy_serials
        ]
        if len(parents) > 1:
            parents.sort(key=lambda serial: distance(hydrogen, atom_by_serial[serial]))
        if parents:
            parent_serial = parents[0]
        else:
            parent = nearest_heavy_parent(hydrogen, heavy_atoms)
            if parent is None:
                raise RestorationError(
                    f"Hydrogen {hydrogen.serial}:{hydrogen.name} has no heavy-atom parent"
                )
            parent_serial = parent.serial
        groups[parent_serial].append(hydrogen)
    for hydrogens in groups.values():
        hydrogens.sort(key=lambda atom: atom.serial)
    return groups


def standard_hydrogen_names(
    parent_name: str, count: int, residue_name: str
) -> list[str]:
    parent_name = parent_name.upper()
    residue_name = residue_name.upper()
    if parent_name == "N":
        return ["H"] if count == 1 else [f"H{index}" for index in range(1, count + 1)]
    if parent_name == "CA":
        if count == 1:
            return ["HA"]
        if residue_name == "GLY" and count == 2:
            return ["HA2", "HA3"]

    locant = parent_name[1:] if len(parent_name) > 1 else parent_name
    base = f"H{locant}"
    if count == 1:
        return [base]
    if count == 2 and not locant[-1:].isdigit():
        return [f"{base}2", f"{base}3"]
    return [f"{base}{index}" for index in range(1, count + 1)]


def cap_hydrogen_names(parent_name: str, count: int) -> list[str]:
    parent_name = parent_name.upper()
    if count == 1:
        return [f"H{parent_name}"]
    locant = parent_name[1:] or parent_name
    base = f"H{locant}"
    return [f"{base}{index}" for index in range(1, count + 1)]


def unique_pdb_names(
    candidates: Sequence[str], used: set[str], fallback_prefix: str = "H"
) -> list[str]:
    result = []
    fallback_index = 1
    for candidate in candidates:
        name = candidate[:4]
        if name and name not in used:
            used.add(name)
            result.append(name)
            continue
        while True:
            fallback = f"{fallback_prefix}{fallback_index}"[:4]
            fallback_index += 1
            if fallback not in used:
                used.add(fallback)
                result.append(fallback)
                break
    return result


def set_histidine_variants(
    identities: dict[int, Identity],
    parent_groups: dict[int, list[PDBAtom]],
) -> None:
    residue_atoms: dict[tuple[int, str], list[int]] = defaultdict(list)
    for serial, identity in identities.items():
        if identity.resname in HISTIDINE_NAMES:
            residue_atoms[identity.residue_key].append(serial)
    for serials in residue_atoms.values():
        names_with_hydrogen = {
            identities[serial].atom_name for serial in serials if parent_groups.get(serial)
        }
        nd1 = "ND1" in names_with_hydrogen
        ne2 = "NE2" in names_with_hydrogen
        if nd1 and ne2:
            variant = "HIP"
        elif nd1:
            variant = "HID"
        elif ne2:
            variant = "HIE"
        else:
            variant = "HIS"
        record = "ATOM" if variant == "HIS" else "HETATM"
        for serial in serials:
            identities[serial].resname = variant
            identities[serial].record = record


def assign_hydrogen_identities(
    identities: dict[int, Identity],
    parent_groups: dict[int, list[PDBAtom]],
) -> None:
    used_by_residue: dict[tuple[int, str], set[str]] = defaultdict(set)
    for identity in identities.values():
        used_by_residue[identity.residue_key].add(identity.atom_name)

    for parent_serial, hydrogens in parent_groups.items():
        parent = identities[parent_serial]
        if parent.resname == "CAP":
            candidates = cap_hydrogen_names(parent.atom_name, len(hydrogens))
            record = "HETATM"
        else:
            candidates = standard_hydrogen_names(
                parent.atom_name, len(hydrogens), parent.resname
            )
            record = parent.record
        names = unique_pdb_names(
            candidates,
            used_by_residue[parent.residue_key],
            fallback_prefix="H",
        )
        for hydrogen, name in zip(hydrogens, names):
            identities[hydrogen.serial] = Identity(
                atom_name=name,
                resname=parent.resname,
                resseq=parent.resseq,
                icode=parent.icode,
                record=record,
                element="H",
            )


def copy_identity(identity: Identity) -> Identity:
    return Identity(
        atom_name=identity.atom_name,
        resname=identity.resname,
        resseq=identity.resseq,
        icode=identity.icode,
        record=identity.record,
        element=identity.element,
    )


def validate_complete_identity_map(
    atoms: Sequence[PDBAtom],
    identities: dict[int, Identity],
    chain: str,
    label: str,
) -> None:
    missing_serials = {atom.serial for atom in atoms} - identities.keys()
    if missing_serials:
        raise RestorationError(
            f"{len(missing_serials)} {label} chain {chain} atom(s) have no "
            "restored identity: "
            + ", ".join(map(str, sorted(missing_serials)[:20]))
        )


def adjacency_bonds_for_atoms(
    atoms: Sequence[PDBAtom], adjacency: dict[int, set[int]]
) -> set[tuple[int, int]]:
    serials = {atom.serial for atom in atoms}
    return {
        tuple(sorted((source, target)))
        for source in serials
        for target in adjacency.get(source, set())
        if target in serials and source != target
    }


def add_hydrogen_parent_bonds(
    bonds: set[tuple[int, int]],
    parent_groups: dict[int, list[PDBAtom]],
) -> None:
    for parent_serial, hydrogens in parent_groups.items():
        for hydrogen in hydrogens:
            bonds.add(tuple(sorted((parent_serial, hydrogen.serial))))


def pdb_atom_name_field(atom_name: str, element: str) -> str:
    atom_name = atom_name.strip()
    if len(atom_name) > 4:
        raise RestorationError(f"Atom name exceeds four PDB columns: {atom_name}")
    if len(atom_name) == 4:
        return atom_name
    if len(element.strip()) == 1 and atom_name and atom_name[0].isalpha():
        return f" {atom_name:<3}"
    return f"{atom_name:>4}"


def rewrite_atom_line(
    atom: PDBAtom, identity: Identity, serial: int
) -> str:
    original = atom.line.rstrip("\r\n")
    if len(original) < 80:
        original = original.ljust(80)
    line = list(original)
    line[0:6] = list(f"{identity.record:<6}"[:6])
    line[6:11] = list(f"{serial:>5d}")
    line[12:16] = list(pdb_atom_name_field(identity.atom_name, identity.element))
    line[17:20] = list(f"{identity.resname:>3}"[-3:])
    line[21:22] = list(atom.chain[:1])
    line[22:26] = list(f"{identity.resseq:>4d}")
    line[26:27] = list((identity.icode or " ")[:1])
    line[76:78] = list(f"{identity.element:>2}"[-2:])
    return "".join(line).rstrip() + "\n"


def rewrite_ter_line(line: str, identity: Identity) -> str:
    raw = line.rstrip("\r\n").ljust(80)
    chars = list(raw)
    chars[17:20] = list(f"{identity.resname:>3}"[-3:])
    chars[22:26] = list(f"{identity.resseq:>4d}")
    chars[26:27] = list((identity.icode or " ")[:1])
    return "".join(chars).rstrip() + "\n"


def rewrite_conect_line(line: str, serial_map: dict[int, int]) -> str:
    values = parse_conect_values(line)
    if not values:
        return line
    mapped = [serial_map.get(value, value) for value in values]
    return "CONECT" + "".join(f"{value:>5d}" for value in mapped) + "\n"


def validate_identity_uniqueness(identities: dict[int, Identity]) -> None:
    names_by_residue: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for identity in identities.values():
        key = (identity.resseq, identity.icode, identity.resname)
        if identity.atom_name in names_by_residue[key]:
            raise RestorationError(
                f"Duplicate atom name {identity.atom_name} in "
                f"{identity.resname} {identity.resseq}{identity.icode}"
            )
        names_by_residue[key].add(identity.atom_name)


def write_restored_pdb(
    input_path: Path,
    output_path: Path,
    lines: Sequence[str],
    relaxed_chain_atoms: Sequence[PDBAtom],
    identities: dict[int, Identity],
    required_bonds: set[tuple[int, int]],
    chain: str,
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        raise RestorationError(f"Output exists (use --overwrite): {output_path}")
    validate_identity_uniqueness(identities)

    ordered_atoms = sorted(
        relaxed_chain_atoms,
        key=lambda atom: (
            identities[atom.serial].resseq,
            identities[atom.serial].icode,
            identities[atom.serial].element in HYDROGEN_ELEMENTS,
            atom.serial,
        ),
    )
    available_serials = sorted(atom.serial for atom in relaxed_chain_atoms)
    serial_map = {
        atom.serial: serial for atom, serial in zip(ordered_atoms, available_serials)
    }
    rewritten_atoms = [
        rewrite_atom_line(atom, identities[atom.serial], serial_map[atom.serial])
        for atom in ordered_atoms
    ]

    chain_line_indexes = {atom.line_index for atom in relaxed_chain_atoms}
    first_chain_index = min(chain_line_indexes)
    last_identity = identities[ordered_atoms[-1].serial]
    output_lines = []
    inserted = False
    for line_index, line in enumerate(lines):
        if line_index in chain_line_indexes:
            if not inserted and line_index == first_chain_index:
                output_lines.extend(rewritten_atoms)
                inserted = True
            continue
        if line.startswith("CONECT"):
            output_lines.append(rewrite_conect_line(line, serial_map))
            continue
        if line[:6].strip() == "TER" and len(line) > 21 and line[21:22].strip() == chain:
            output_lines.append(rewrite_ter_line(line, last_identity))
            continue
        output_lines.append(line)
    if not inserted:
        raise RestorationError(f"{input_path}: failed to insert restored chain {chain}")

    existing_adjacency = conect_adjacency(lines)
    existing_bonds = {
        tuple(sorted((source, target)))
        for source, targets in existing_adjacency.items()
        for target in targets
        if source != target
    }
    missing_bonds = sorted(required_bonds - existing_bonds)
    additional_conect = [
        f"CONECT{serial_map.get(source, source):>5d}"
        f"{serial_map.get(target, target):>5d}\n"
        for source, target in missing_bonds
    ]
    if additional_conect:
        end_index = next(
            (
                index
                for index, line in enumerate(output_lines)
                if line.strip().upper() in {"END", "ENDMDL"}
            ),
            len(output_lines),
        )
        output_lines[end_index:end_index] = additional_conect

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.writelines(output_lines)
    try:
        os.replace(temporary_path, output_path)
        os.chmod(output_path, input_path.stat().st_mode & 0o777)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    _, output_atoms = read_pdb(output_path)
    output_by_serial = {atom.serial: atom for atom in output_atoms}
    for input_atom in relaxed_chain_atoms:
        output_atom = output_by_serial.get(serial_map[input_atom.serial])
        if output_atom is None:
            raise RestorationError(
                f"Post-write validation lost atom {input_atom.serial}"
            )
        if input_atom.coord != output_atom.coord:
            raise RestorationError(
                f"Coordinate changed for atom {input_atom.serial}: "
                f"{input_atom.coord} -> {output_atom.coord}"
            )
    return len(missing_bonds)


def restore_one(
    relaxed_path: Path,
    metadata: MetadataEntry,
    raw_path: Path,
    merged_path: Path,
    output_path: Path,
    accepted_output_path: Path | None,
    chain: str,
    tolerance: float,
    overwrite: bool,
) -> RestoredResult:
    relaxed_lines, relaxed_all_atoms = read_pdb(relaxed_path)
    merged_lines, merged_all_atoms = read_pdb(merged_path)

    relaxed_chain_atoms = [
        atom for atom in relaxed_all_atoms if atom.chain == chain
    ]
    merged_chain_atoms = [
        atom for atom in merged_all_atoms if atom.chain == chain
    ]
    relaxed_heavy = [atom for atom in relaxed_chain_atoms if not atom.is_hydrogen]
    merged_heavy = [atom for atom in merged_chain_atoms if not atom.is_hydrogen]
    raw_peptide, raw_placeholders, raw_format = load_raw_peptide_atoms(
        raw_path, metadata, chain
    )
    if not relaxed_chain_atoms:
        raise RestorationError(f"{relaxed_path}: no atoms in chain {chain}")
    if not merged_heavy:
        raise RestorationError(f"{merged_path}: no heavy atoms in chain {chain}")

    merged_adjacency = conect_adjacency(merged_lines)
    relaxed_adjacency = conect_adjacency(relaxed_lines)
    raw_to_merged = coordinate_mapping(raw_peptide, merged_heavy, tolerance)
    validate_peptide_mapping(
        raw_peptide, merged_heavy, raw_to_merged, merged_adjacency
    )
    merged_to_relaxed = merged_to_relaxed_mapping(
        merged_heavy,
        relaxed_heavy,
        merged_adjacency,
        relaxed_adjacency,
    )

    identities: dict[int, Identity] = {}
    peptide_by_merged_index = {}
    for raw_index, merged_index in raw_to_merged.items():
        raw_atom = raw_peptide[raw_index]
        relaxed_index = merged_to_relaxed[merged_index]
        relaxed_atom = relaxed_heavy[relaxed_index]
        peptide_by_merged_index[merged_index] = raw_atom
        identities[relaxed_atom.serial] = Identity(
            atom_name=raw_atom.name,
            resname=raw_atom.resname,
            resseq=raw_atom.resseq,
            icode=raw_atom.icode,
            record="ATOM",
            element=raw_atom.element,
        )

    cap_indexes = [
        index
        for index in range(len(merged_heavy))
        if index not in peptide_by_merged_index
    ]
    if not cap_indexes:
        raise RestorationError("Merged structure contains no CAP heavy atoms")
    cap_names, crosslinks = generic_cap_names(
        cap_indexes,
        merged_heavy,
        peptide_by_merged_index,
        merged_adjacency,
    )
    peptide_residue_numbers = {atom.resseq for atom in raw_peptide}
    placeholder_numbers = {
        atom.resseq
        for atom in raw_placeholders
        if atom.resseq not in peptide_residue_numbers
    }
    cap_resseq = (
        min(placeholder_numbers)
        if placeholder_numbers
        else min(peptide_residue_numbers) - 1
    )
    for merged_index in cap_indexes:
        relaxed_index = merged_to_relaxed[merged_index]
        relaxed_atom = relaxed_heavy[relaxed_index]
        identities[relaxed_atom.serial] = Identity(
            atom_name=cap_names[merged_index],
            resname="CAP",
            resseq=cap_resseq,
            icode="",
            record="HETATM",
            element=relaxed_atom.element,
        )

    parent_groups = hydrogen_parent_groups(
        relaxed_chain_atoms, identities, relaxed_adjacency
    )
    set_histidine_variants(identities, parent_groups)
    assign_hydrogen_identities(identities, parent_groups)
    validate_complete_identity_map(
        relaxed_chain_atoms, identities, chain, "relaxed"
    )

    required_bonds = set()
    merged_serial_to_index = {
        atom.serial: index for index, atom in enumerate(merged_heavy)
    }
    for merged_index, merged_atom in enumerate(merged_heavy):
        for neighbor_serial in merged_adjacency.get(merged_atom.serial, set()):
            neighbor_index = merged_serial_to_index.get(neighbor_serial)
            if neighbor_index is None or neighbor_index <= merged_index:
                continue
            relaxed_atom1 = relaxed_heavy[merged_to_relaxed[merged_index]]
            relaxed_atom2 = relaxed_heavy[merged_to_relaxed[neighbor_index]]
            required_bonds.add(
                tuple(sorted((relaxed_atom1.serial, relaxed_atom2.serial)))
            )
    add_hydrogen_parent_bonds(required_bonds, parent_groups)

    added_conect_bonds = write_restored_pdb(
        relaxed_path,
        output_path,
        relaxed_lines,
        relaxed_chain_atoms,
        identities,
        required_bonds,
        chain,
        overwrite,
    )

    accepted_added_conect_bonds = 0
    accepted_coordinate_check = "not_requested"
    if accepted_output_path is not None:
        accepted_identities = {}
        for merged_index, merged_atom in enumerate(merged_heavy):
            relaxed_index = merged_to_relaxed[merged_index]
            relaxed_atom = relaxed_heavy[relaxed_index]
            accepted_identities[merged_atom.serial] = copy_identity(
                identities[relaxed_atom.serial]
            )
        accepted_parent_groups = hydrogen_parent_groups(
            merged_chain_atoms,
            accepted_identities,
            merged_adjacency,
        )
        assign_hydrogen_identities(
            accepted_identities,
            accepted_parent_groups,
        )
        validate_complete_identity_map(
            merged_chain_atoms,
            accepted_identities,
            chain,
            "accepted",
        )
        accepted_required_bonds = adjacency_bonds_for_atoms(
            merged_chain_atoms,
            merged_adjacency,
        )
        add_hydrogen_parent_bonds(
            accepted_required_bonds,
            accepted_parent_groups,
        )
        accepted_added_conect_bonds = write_restored_pdb(
            merged_path,
            accepted_output_path,
            merged_lines,
            merged_chain_atoms,
            accepted_identities,
            accepted_required_bonds,
            chain,
            overwrite,
        )
        accepted_coordinate_check = "exact"

    match_distances = [
        distance(raw_peptide[raw_index], merged_heavy[merged_index])
        for raw_index, merged_index in raw_to_merged.items()
    ]
    residues_by_number = {}
    for identity in identities.values():
        residues_by_number[(identity.resseq, identity.icode)] = identity.resname
    residues = [
        f"{name}{resseq}{icode}"
        for (resseq, icode), name in sorted(residues_by_number.items())
    ]
    return RestoredResult(
        relaxed=relaxed_path,
        metadata=metadata.path,
        raw=raw_path,
        merged=merged_path,
        output=output_path,
        accepted_output=accepted_output_path,
        peptide_heavy_atoms=len(raw_peptide),
        cap_heavy_atoms=len(cap_indexes),
        hydrogen_atoms=sum(atom.is_hydrogen for atom in relaxed_chain_atoms),
        residues=residues,
        raw_merged_rmsd=math.sqrt(
            sum(value * value for value in match_distances) / len(match_distances)
        ),
        raw_merged_max_distance=max(match_distances),
        crosslink_count=len(crosslinks),
        added_conect_bonds=added_conect_bonds,
        accepted_added_conect_bonds=accepted_added_conect_bonds,
        coordinate_check="exact",
        accepted_coordinate_check=accepted_coordinate_check,
        raw_format=raw_format,
    )


def relaxed_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob(pattern))
    if not files:
        raise RestorationError(f"No relaxed PDB files match {pattern!r} under {path}")
    return files


def output_path_for(
    relaxed_path: Path, relaxed_root: Path, output_root: Path
) -> Path:
    if relaxed_root.is_file():
        relative = Path(relaxed_path.name)
    else:
        relative = relaxed_path.relative_to(relaxed_root)
    stem = re.sub(r"_restored$", "", relative.stem)
    return output_root / relative.with_name(f"{stem}_restored.pdb")


def accepted_output_path_for(
    relaxed_path: Path,
    relaxed_root: Path,
    accepted_output_root: Path,
) -> Path:
    if relaxed_root.is_file():
        relative = Path(relaxed_path.name)
    else:
        relative = relaxed_path.relative_to(relaxed_root)
    stem = re.sub(r"_relaxed$", "", relative.stem)
    stem = re.sub(r"_restored$", "", stem)
    return accepted_output_root / relative.with_name(f"{stem}_restored.pdb")


def execute_restoration_job(job: RestorationJob) -> RestoredResult:
    return restore_one(
        relaxed_path=job.relaxed_path,
        metadata=job.metadata,
        raw_path=job.raw_path,
        merged_path=job.merged_path,
        output_path=job.output_path,
        accepted_output_path=job.accepted_output_path,
        chain=job.chain,
        tolerance=job.tolerance,
        overwrite=job.overwrite,
    )


def failure_record(relaxed_path: Path, error: Exception) -> dict:
    return {
        "status": "failed",
        "relaxed": str(relaxed_path),
        "error": str(error),
    }


def effective_worker_count(requested: int, job_count: int) -> int:
    if requested < 0:
        raise RestorationError("--workers must be 0 or a positive integer")
    configured = min(8, os.cpu_count() or 1) if requested == 0 else requested
    return max(1, min(configured, max(1, job_count)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore chain-L CAP/amino-acid identities in flattened relaxed PDBs "
            "without changing coordinates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--relaxed-dir",
        type=Path,
        required=True,
        help="Relaxed PDB file or folder to scan recursively.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Metadata JSON file or accepted-library metadata folder.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Output root. The input directory tree is preserved below it; "
            "defaults to <relaxed-dir>_restored."
        ),
    )
    parser.add_argument(
        "--accepted-output-dir",
        type=Path,
        help=(
            "Optional output root for the unrelaxed accepted PDBs restored "
            "with exactly the same heavy-atom names as the relaxed outputs."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help=(
            "Optional relocated raw root. PepAny/CPcomposer PDB and BoltzGen "
            "CIF/CIF.GZ inputs are supported."
        ),
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        help="Optional relocated accepted complexes root when JSON paths are stale.",
    )
    parser.add_argument(
        "--pattern",
        default="*_relaxed.pdb",
        help="Recursive filename glob used inside --relaxed-dir.",
    )
    parser.add_argument("--chain", default="L", help="Flattened ligand chain ID.")
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=2.0,
        help="Maximum raw→unrelaxed merged atom-match distance in Å.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes; 0 selects min(8, available CPUs), 1 is serial.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing *_restored.pdb outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    relaxed_root = args.relaxed_dir.expanduser().resolve()
    metadata_root = args.metadata_dir.expanduser().resolve()
    raw_root = args.raw_dir.expanduser().resolve() if args.raw_dir else None
    merged_root = (
        args.merged_dir.expanduser().resolve() if args.merged_dir else None
    )
    if not relaxed_root.exists():
        raise RestorationError(f"--relaxed-dir does not exist: {relaxed_root}")
    if not metadata_root.exists():
        raise RestorationError(f"--metadata-dir does not exist: {metadata_root}")
    if raw_root is not None and not raw_root.exists():
        raise RestorationError(f"--raw-dir does not exist: {raw_root}")
    if merged_root is not None and not merged_root.exists():
        raise RestorationError(f"--merged-dir does not exist: {merged_root}")

    if args.output_dir:
        output_root = args.output_dir.expanduser().resolve()
    elif relaxed_root.is_file():
        output_root = relaxed_root.parent / f"{relaxed_root.stem}_restored"
    else:
        output_root = relaxed_root.with_name(f"{relaxed_root.name}_restored")
    accepted_output_root = (
        args.accepted_output_dir.expanduser().resolve()
        if args.accepted_output_dir
        else None
    )
    if accepted_output_root == output_root:
        raise RestorationError(
            "--accepted-output-dir and --output-dir must be different folders"
        )

    entries = load_metadata_entries(metadata_root)
    inputs = relaxed_files(relaxed_root, args.pattern)
    entries_by_sample: dict[int, list[MetadataEntry]] = defaultdict(list)
    for entry in entries:
        entries_by_sample[entry.sample_index].append(entry)

    source_path_cache: dict[tuple[Path, str], list[Path]] = {}
    jobs: list[RestorationJob] = []
    records: list[dict | None] = [None] * len(inputs)
    preparing = track_progress(
        inputs,
        total=len(inputs),
        description="Preparing jobs",
        unit="file",
    )
    for position, relaxed_path in enumerate(preparing):
        try:
            sample = sample_number_from_path(relaxed_path)
            candidate_metadata = entries_by_sample.get(sample, [])
            metadata = match_metadata(relaxed_path, candidate_metadata)
            raw_path = resolve_source_pdb(
                metadata,
                "raw",
                relaxed_path,
                metadata_root,
                raw_root,
                source_path_cache,
            )
            merged_path = resolve_source_pdb(
                metadata,
                "merged",
                relaxed_path,
                metadata_root,
                merged_root,
                source_path_cache,
            )
            accepted_output_path = (
                accepted_output_path_for(
                    relaxed_path,
                    relaxed_root,
                    accepted_output_root,
                )
                if accepted_output_root is not None
                else None
            )
            if (
                accepted_output_path is not None
                and accepted_output_path.resolve() == merged_path.resolve()
            ):
                raise RestorationError(
                    f"Accepted output would overwrite its input: {merged_path}"
                )
            jobs.append(
                RestorationJob(
                    position=position,
                    relaxed_path=relaxed_path,
                    metadata=metadata,
                    raw_path=raw_path,
                    merged_path=merged_path,
                    output_path=output_path_for(
                        relaxed_path, relaxed_root, output_root
                    ),
                    accepted_output_path=accepted_output_path,
                    chain=args.chain,
                    tolerance=args.coordinate_tolerance,
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:
            records[position] = failure_record(relaxed_path, exc)
            progress_write(f"[prepare] ERROR {relaxed_path}: {exc}", error=True)

    workers = effective_worker_count(args.workers, len(jobs))
    report = {
        "format_version": 1,
        "relaxed_root": str(relaxed_root),
        "metadata_root": str(metadata_root),
        "output_root": str(output_root),
        "accepted_output_root": (
            str(accepted_output_root) if accepted_output_root else None
        ),
        "workers": workers,
        "success_count": 0,
        "failure_count": 0,
        "files": [],
    }

    progress_write(
        f"Prepared {len(jobs)}/{len(inputs)} job(s); "
        f"running with {workers} worker process(es)."
    )
    completed_success = 0
    completed_failure = len(inputs) - len(jobs)
    restoring = progress_bar(
        total=len(jobs),
        description="Restoring",
        unit="file",
    )

    def store_result(
        job: RestorationJob,
        result: RestoredResult | None = None,
        error: Exception | None = None,
    ) -> None:
        nonlocal completed_success, completed_failure
        if error is not None:
            completed_failure += 1
            records[job.position] = failure_record(job.relaxed_path, error)
            progress_write(
                f"ERROR {job.relaxed_path}: {error}",
                error=True,
            )
        else:
            if result is None:
                raise RestorationError("Internal error: job returned no result")
            completed_success += 1
            records[job.position] = result.as_dict()
        restoring.update()
        restoring.set_postfix(
            {
                "ok": completed_success,
                "failed": completed_failure,
            }
        )

    try:
        if workers == 1:
            for job in jobs:
                try:
                    store_result(job, result=execute_restoration_job(job))
                except Exception as exc:
                    store_result(job, error=exc)
        elif jobs:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_job = {
                    executor.submit(execute_restoration_job, job): job for job in jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        store_result(job, result=future.result())
                    except Exception as exc:
                        store_result(job, error=exc)
    finally:
        restoring.close()

    report["files"] = [record for record in records if record is not None]
    report["success_count"] = sum(
        record["status"] == "success" for record in report["files"]
    )
    report["failure_count"] = sum(
        record["status"] == "failed" for record in report["files"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "restoration_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    accepted_report_path = None
    if accepted_output_root is not None:
        accepted_output_root.mkdir(parents=True, exist_ok=True)
        accepted_report_path = accepted_output_root / "restoration_report.json"
        accepted_report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    print(
        f"Finished: {report['success_count']} succeeded, "
        f"{report['failure_count']} failed. Report: {report_path}"
        + (
            f"; accepted report: {accepted_report_path}"
            if accepted_report_path is not None
            else ""
        )
    )
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RestorationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
