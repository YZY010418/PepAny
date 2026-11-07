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

The public [Zenodo record](https://zenodo.org/records/20080102) distributes
`pepany_v1.ckpt` (423,133,434 bytes) under **CC BY 4.0**. Its published MD5 is
`3618257f08b99d1aac685247ed00af73`. The code license is separately MIT; see [LICENSE](LICENSE).
Pass the downloaded checkpoint to generation with `--ckpt`.

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

CPcap is publicly available in the [CPSea2 dataset on Hugging Face](https://huggingface.co/datasets/YZY010418/CPSea2/tree/main/CPcap)
under **CC BY 4.0**. It is distributed separately from this repository and the
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

## Outputs

Each template creates a subdirectory under `--save_dir`. `candidates/` contains
sampled structures and generation statistics; with cyclic assembly enabled,
`accepted/` contains structures that pass cap alignment and clash filtering.

## Contact

For questions or feedback, please open a [GitHub issue](https://github.com/YZY010418/PepAny/issues)
or contact Ziyi Yang at [yangzy23@mails.tsinghua.edu.cn](mailto:yangzy23@mails.tsinghua.edu.cn).
