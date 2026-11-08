# PepAny: Generating cyclic peptide binders with any cyclization topologies

**PepAny** designs peptide binders with classifier-free guidance (CFG) over
terminal amide geometry. Given a terminal-cap conformer, it samples peptides
that can be assembled with the cap into cyclic binders.

PepAny is an application of [CPSea-2](https://anonymous.4open.science/r/CPSea2-6CF5).

![PepAny overview](assets/cover.png)

## Installation

PepAny is tested with Python 3.9 and PyTorch. Choose the conda environment file that matches your CUDA version:

```bash
conda env create -f env_cuda121.yaml
conda activate PepAny
```

Download the pretrained weights:

```bash
mkdir -p checkpoints
wget -c "https://zenodo.org/records/20080102/files/pepany_v1.ckpt?download=1" -O checkpoints/pepany_v1.ckpt
```

## Quick Start

Run all commands from the repository root. Extract the bundled target examples,
then run the linear-peptide example (no terminal-cap library required):

```bash
tar -xzf api/CPtest.tar.gz -C api
python -m api.generate \
  --config api/demo/linear.yaml \
  --ckpt checkpoints/pepany_v1.ckpt \
  --save_dir outputs/demo \
  --gpu 0
```

### Cyclic peptides: download CPcap

CPcap is publicly available in the [CPSea2 dataset on Hugging Face](https://huggingface.co/datasets/YZY010418/CPSea2/tree/main/CPcap). It is distributed separately from this repository and the
PepAny checkpoint. Download the eight conformer archives used by
`api/demo/config.yaml` and extract them into the paths expected by that config.
Run this Bash block from the repository root:

```bash
set -e
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
for topology in Alkene-A Amide-A Disulfide-A Ester-A ThioEther-A ThioEther-B ThioEther-C Triazole-A; do
  archive="api/CPcap/CPcap_${topology}_Conformers.tar.gz"
  destination="api/CPcap/${topology}/accepted_energy_filtered_sdf"
  mkdir -p "$destination"
  curl --fail --location --retry 3 \
    "${HF_ENDPOINT}/datasets/YZY010418/CPSea2/resolve/main/CPcap/CPcap_${topology}/CPcap_${topology}_Conformers.tar.gz" \
    --output "$archive"
  tar -xzf "$archive" --strip-components=1 -C "$destination"
done
```

The command places SDF files in the directories expected by the demo config.
CPcap data is ignored by Git and is not included in this repository. If needed,
set `HF_ENDPOINT=https://hf-mirror.com` before running the download block.

After downloading the checkpoint and extracting the target examples as above:

```bash
python -m api.generate \
  --config api/demo/config.yaml \
  --ckpt checkpoints/pepany_v1.ckpt \
  --save_dir outputs/cyclic_demo \
  --gpu 0
```

Set a finite `max_sampling_number` in each template for bounded runs; the demo's
`null` budget continues until its acceptance goal is reached. For custom cap
libraries, update both the SDF paths and terminal atom indices in the config.

## Use your own target

Copy [the linear config](api/demo/linear.yaml) or [the cyclic config](api/demo/config.yaml)
and pass your edited file with `--config`.

- **Target and pocket:** set `dataset.pdb_path`, `tgt_chains`, and `lig_chains`.
  Multiple chains can be specified as `[A, B, C]`. `pocket_threshold` selects
  receptor residues near the reference ligand using CA distances in angstroms.
- **Target without a reference ligand:** omit `lig_chains`, set
  `generated_lig_chain: L`, and specify `size_min` and `size_max`. All selected
  target residues are used unless you provide `pocket_residues`, for example
  `[[R, 42, ""], [R, 43, ""]]` (chain, residue number, insertion code).
- **Peptide length:** `size_min` is inclusive and `size_max` is exclusive;
  `8` and `13` sample 8–12 core residues. With a reference ligand, you may use
  `size_from_ligand: true` instead.
- **Terminal caps:** `capping: true` uses ACE/NME terminal caps;
  `capping: false` generates uncapped linear peptides and skips cyclic assembly.
- **Cyclic assembly:** `term_rel_pos_sdf` specifies cap conformers and terminal
  atom indices. `pipeline.goal_num` controls accepted structures per target;
  `max_sampling_number` limits attempts. Set `pipeline.enabled: false` to
  generate candidates without cap assembly and acceptance filtering.
- **Sampling:** use `n_samples`, `batch_size`, and `seed` to control the run.

### Pocket configuration examples

Replace the `dataset` block in a demo config with one of the following.
Paths and residue numbers below are placeholders for your own structures.

**1. Define the pocket around a reference peptide.** Select receptor residues
within `pocket_threshold` angstroms of the reference ligand (CA distances):

```yaml
dataset:
  pdb_path: inputs/target_with_reference.pdb
  tgt_chains: R                 # or [A, B] for a multichain receptor
  lig_chains: L
  pocket_threshold: 10.0
```

**2. Specify pocket residues directly, without a reference peptide.** Residues
are `[chain_id, residue_number, insertion_code]` triples. This selects exactly
the listed receptor residues; include the surrounding context needed for design.

```yaml
dataset:
  pdb_path: inputs/target.pdb
  tgt_chains: R
  generated_lig_chain: L
  pocket_residues:
    - [R, 42, ""]
    - [R, 43, ""]
    - [R, 57, A]
```

Without `lig_chains`, use explicit peptide length bounds in the template:

```yaml
templates:
  - name: custom_pocket
    class: Peptide
    capping: true
    size_min: 8
    size_max: 13
n_samples: 100
batch_size: 10
seed: 12
```

Omit `pocket_residues` to use all selected receptor-chain residues as context.
Do not use `size_from_ligand: true` when no reference ligand is supplied.

**3. Specify different pockets for multiple targets using a TSV.**

```yaml
dataset:
  pdb_paths:
    - inputs/target_a.pdb
    - inputs/target_b.pdb
  tgt_chains: R
  generated_lig_chain: L
  pocket_residues_tsv: inputs/pockets.tsv
```

Use one residue per row in `inputs/pockets.tsv` (tab-separated). File names may
match the input path, basename, or stem; `.` denotes an empty insertion code.
Use the same explicit-length template as above.

```tsv
pdb_file	chain_id	res_id	res_ins_code
target_a.pdb	R	42	.
target_a.pdb	R	43	.
target_b.pdb	R	57	A
```

### No reference peptide: dummy → decoys → reference

As an alternative to selecting residues directly, manually place a dummy
peptide in the intended binding site in a molecular editor. Save the receptor
as chain `R` and the dummy as a separate chain `L` in the same PDB coordinate
frame. Use standard peptide residues with valid backbone atoms (including CA),
not an arbitrary point or an unrecognized dummy atom. The dummy locates the
pocket; it is not evidence of binding.

Save this first-pass config as `configs/dummy_decoys.yaml`:

```yaml
dataset:
  pdb_path: inputs/target_with_dummy.pdb
  tgt_chains: R
  lig_chains: L
  pocket_threshold: 10.0

templates:
  - name: dummy_decoys
    class: Peptide
    capping: true
    size_min: 8
    size_max: 13
    pipeline:
      enabled: false

n_samples: 100
batch_size: 10
seed: 12
```

```bash
python -m api.generate \
  --config configs/dummy_decoys.yaml \
  --ckpt checkpoints/pepany_v1.ckpt \
  --save_dir outputs/dummy_decoys \
  --gpu 0
```

Inspect the resulting candidates and select a plausible decoy at the intended
site, considering geometry, clashes, and receptor contacts. `n_samples: 100`
requests about 100 decoys; invalid samples may reduce the number of usable
structures. Selection is manual, not an automatic affinity ranking.

Prepare `inputs/target_with_selected_decoy.pdb` containing the receptor and the
selected decoy as chain `L`, preserving their relative coordinates and removing
the original dummy. Use that decoy as the reference peptide for a second run:

```yaml
dataset:
  pdb_path: inputs/target_with_selected_decoy.pdb
  tgt_chains: R
  lig_chains: L
  pocket_threshold: 10.0
```

Use this dataset block in a copy of `api/demo/config.yaml` for cyclic generation
(after downloading CPcap), or in the linear config for further linear sampling.
The cyclic templates can now use `size_from_ligand: true`. Run with a new
`--save_dir`, such as `outputs/refined_designs`. The selected decoy serves as a
structural reference, not an experimentally validated positive binder.

## Outputs

Each template creates a subdirectory under `--save_dir`. `candidates/` contains
sampled structures and generation statistics; with cyclic assembly enabled,
`accepted/` contains structures that pass cap alignment and clash filtering.

## Contact

For questions or feedback, please open a [GitHub issue](https://github.com/YZY010418/PepAny/issues)
or contact Ziyi Yang at [yangzy23@mails.tsinghua.edu.cn](mailto:yangzy23@mails.tsinghua.edu.cn).
