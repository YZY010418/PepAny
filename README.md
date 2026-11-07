# PepAny: Generating cyclic peptide binders with any cyclization topologies

**PepAny** is a generative model for peptide binder design that allows CFG control over the terminal amides. Given a capping structure, PepAny can automatically read the terminal constraints, and attempts to sample peptide binders that can merge with the capping structure and produce cyclic peptide binders.

This model is an example application of [**CPSea-2**](https://anonymous.4open.science/r/CPSea2-6CF5).

![PepAny overview](assets/cover.png)


## Installation

PepAny is tested with Python 3.9 and PyTorch. Choose the conda environment file that matches your CUDA version:

```bash
conda env create -f env_cuda121.yaml
conda activate PepAny
```

Please download our pretrained weights by

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

### Download CPcap for cyclic peptide generation

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

The archives contain a top-level `CPcap_<topology>_Conformers/` directory;
`--strip-components=1` places the SDF files directly in the configured directory,
for example `api/CPcap/Amide-A/accepted_energy_filtered_sdf/*.sdf`.
The downloaded library is ignored by Git. If Hugging Face is unreachable from
your network, you can set `HF_ENDPOINT` to an accessible mirror before running
the block (for example, `export HF_ENDPOINT=https://hf-mirror.com`).

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

Training configurations under `configs/` are templates. Their
`datasets/CPSea2.0_PDB` paths require separately prepared training data.

## Input Config

Generation is controlled by a YAML file. A minimal configuration for linear peptide binders looks like:

```yaml
dataset:
  pdb_path: ./api/CPtest/B_G_pdb4gba_renamed.pdb
  tgt_chains: R # receptor chains; use [A, B, C] for multiple chains
  lig_chains: L # reference ligand chains defining the pocket; use [A, B, C] for multiple chains
  pocket_threshold: 10.0 # receptor residues within this distance from ligand are used as pocket

templates:
  - name: linear_peptide_capped
    class: Peptide
    capping: true
    size_min: 8
    size_max: 13
  - name: linear_peptide_uncapped
    class: Peptide
    capping: false
    size_from_ligand: true

n_samples: 10
seed: 12
batch_size: 10
```

* `size_min` is inclusive and `size_max` is exclusive, so the example samples 8-12 core peptide residues. You can also use `size_from_ligand` where the generated peptides will be of the same length as the ligand reference.
* By default the model generate linear peptides with ACE at N terminal and NME at C terminal as capping structures. You can choose not to include these capping structures by setting `capping: false`
* `pocket_threshold` controls the automatic receptor pocket cutoff in Angstroms based on CA distances.

**If your input PDBs contain only the target chain**, omit `lig_chains`. PepAny will:

(1) use all selected target-chain residues as context and generate a new peptide chain named by `generated_lig_chain`:

```yaml
dataset:
  pdb_path: ./api/target_only/target.pdb
  tgt_chains: R
  generated_lig_chain: L

templates:
  - name: linear_peptide_capped
    class: Peptide
    capping: true
    size_min: 8
    size_max: 13
  - name: linear_peptide_uncapped
    class: Peptide
    capping: false
    size_min: 8
    size_max: 13

n_samples: 10
seed: 12
batch_size: 10
```

(2) use residues that are manually appointed by `pocket_residues`, which is supposed to be a list of `(chainID, resID, res_ins_code)` tuples:

```yaml
dataset:
  pdb_path: ./api/target_only/target.pdb
  tgt_chains: R
  generated_lig_chain: L
  pocket_residues:
    - [R, 42, ""]
    - [R, 43, ""]
    - [R, 57, A]
```

For multiple input files, put the pocket residues in a TSV and point `pocket_residues_tsv` to it:

```yaml
dataset:
  pdb_paths:
    - ./api/target_only/target_a.pdb
    - ./api/target_only/target_b.pdb
  tgt_chains: R
  generated_lig_chain: L
  pocket_residues_tsv: ./api/target_only/pockets.tsv
```

The TSV must have one residue per row. File names can match the full path, basename, or stem:

```tsv
pdb_file	chain_id	res_id	res_ins_code
target_a.pdb	R	42	.
target_a.pdb	R	43	.
target_b.pdb	R	57	A
```

**For cyclic peptide generation**, we add additional conditions in template that define a cyclization topology and an SDF library of terminal cap conformers:

```yaml
templates:
  - name: ThioEther-C
    class: Peptide
    capping: true
    size_from_ligand: true
    term_rel_pos_sdf:
      path: ./api/CPcap/ThioEther-C/accepted_energy_filtered_sdf
      pattern: "*.sdf"
      sampling_mode: with_replacement
      index_base: 0
      n_term_atom_indices: [9, 10, 13, 11, 12]
      c_term_atom_indices: [0, 1, 29, 2, 3]
    pipeline:
      enabled: true
      goal_num: 10
      max_sampling_number: 100000
      rmsd_cutoff: 1.0
      clash_threshold: 0.8
      output_dir: accepted
```

Important fields:

- `goal_num`: number of accepted structures required *for each target*.
- `max_sampling_number`: maximum number of candidate generations before stopping. Use `null` for no fixed budget (i.e., infinite sampling until satisfying the goal_num).
- `sampling_mode`: with_replacement means terminal conformers may be reused. without_replacement means terminal conformers are shuffled and consumed without reuse.
- `rmsd_cutoff`: terminal cap alignment cutoff used by the accepted pipeline.
- `clash_threshold`: receptor clash distance threshold.
- `seed`: random seed. It can also be overridden by `--seed`.

If you want to use only one terminal-cap SDF instead of a directory of SDF files, set `term_rel_pos_sdf.path` directly to that file and omit `pattern`:

```yaml
templates:
  - name: ThioEther-C_single_sdf
    class: Peptide
    capping: true
    size_from_ligand: true
    term_rel_pos_sdf:
      path: ./api/CPcap/ThioEther-C/example_cap.sdf
      index_base: 0
      n_term_atom_indices: [9, 10, 13, 11, 12]
      c_term_atom_indices: [0, 1, 29, 2, 3]
    pipeline:
      enabled: true
      goal_num: 10
      max_sampling_number: 100000
```

If you do not want to use the automatic pipeline, omit the `pipeline` block or set `pipeline.enabled: false`. PepAny will still use `term_rel_pos_sdf` as a terminal geometry condition, but it will only write raw candidates under `candidates/`; it will not merge the cap, run RMSD/clash filtering, or write an `accepted/` directory. In this mode, use `n_samples` to control how many candidates are generated:

```yaml
templates:
  - name: ThioEther-C_raw_candidates
    class: Peptide
    capping: true
    size_from_ligand: true
    term_rel_pos_sdf:
      path: ./api/CPcap/ThioEther-C/example_cap.sdf
      index_base: 0
      n_term_atom_indices: [9, 10, 13, 11, 12]
      c_term_atom_indices: [0, 1, 29, 2, 3]
    pipeline:
      enabled: false

n_samples: 100
```

Note: When `capping: false`, PepAny skips the cyclic accepted pipeline even if `pipeline.enabled` is set, because terminal cap merging requires the capped scaffold.

### Controlled raw sampling

For a matched-budget comparison of the generation model itself, use the dedicated
CPtest wrapper. It defaults to 100 model attempts per target and cyclization type,
CFG scale 2.5, and a 15 Angstrom pocket cutoff:

```bash
python scripts/experiments/sample_cptest_raw.py \
  --config api/demo/config.yaml \
  --ckpt checkpoints/pepany_v1.ckpt \
  --save-dir outputs/cptest_raw \
  --gpu 0
```

This mode performs exactly 100 model attempts rather than sampling until 100
candidates pass. It disables the accepted pipeline, template validation, amide
checks, terminal RMSD calculation, clash checks, cap merging, relaxation, and
affinity scoring. Each target directory contains:

- `target.pdb`: the fixed receptor conformer;
- `raw_model_outputs/<sample>.pt`: the literal sampled tensors and metadata;
- `raw_model_outputs/<sample>.json`: human-readable provenance;
- `conditioning_conformers/<sample>.sdf`: the terminal-conditioning conformer;
- `<sample>.pdb` and `<sample>.sdf`: best-effort structural serialization of the
  raw sample, with no acceptance filter.

If structural serialization fails, its `.pt` and `.json` files remain and the
attempt is not replaced. Use a new output directory for each controlled run; the
wrapper refuses to mix a run with existing files.

Evaluate the transferred raw attempts on a CPU node with:

```bash
python scripts/experiments/evaluate_pepany_raw_geometry.py \
  --input outputs/cptest_raw \
  --workers 0 \
  --expected-attempts-per-target 100
```

The default 1 Angstrom cutoffs are applied separately to the absolute terminal
CA-distance error and the Kabsch-aligned terminal 10-atom RMSD. Reports are
written under `<input>/geometry_evaluation`; no structures are copied by
default. Each raw metadata JSON remains in the success-rate denominator even if
its candidate SDF is missing or invalid. The evaluator is self-contained and
does not import other PepAny modules; it only requires Python, NumPy, and RDKit
(`tqdm` is optional).

To additionally compare every raw candidate with every capping conformer of its
cyclization type, point the evaluator to the CPcap root:

```bash
python scripts/experiments/evaluate_pepany_raw_geometry.py \
  --input outputs/cptest_raw \
  --capping-library-root api/CPcap \
  --workers 0
```

For each run name, the evaluator recursively reads SDFs below
`<capping-library-root>/<run_name>`. A custom mapping can be supplied when a run
name and library directory differ:

```bash
--capping-library Amide-A=/path/to/amide/conformers \
--capping-library Disulfide-A=/path/to/disulfide/conformers
```

The library results separately report whether any conformer passes the CA
criterion, whether any conformer passes the 10-atom RMSD criterion, and whether
one same conformer passes both. Conformers below the 10-atom RMSD cutoff are
tried in increasing-RMSD order: the evaluator aligns and merges each CPcap
structure into the raw peptide and checks nonlocal heavy-atom clashes, stopping
as soon as it finds a clash-free option. An attempt passes
`library_align_no_clash_pass` when at least one RMSD-passing conformer produces a
clash-free merge. By default, a clash is a distance at or below
`0.8 * (vdw_radius_1 + vdw_radius_2)`; change the scale with
`--clash-threshold`. Atom pairs separated by at most three bonds are excluded.
The summary reports both the align+no-clash rate over all raw attempts and the
no-clash rate among attempts that passed the any-library RMSD criterion. The
per-attempt report also records the minimum errors, conformer counts, merge
errors, and best-matching conformer paths.

For success-rate and failure-mode breakdowns by cyclization type and generated
peptide length, add:

```bash
--grouped-success-rates
```

This writes `per_cyclization_summary.csv`, `per_length_summary.csv`, and
`per_cyclization_length_summary.csv`. Peptide length is the number of generated
core amino-acid residues after excluding the synthetic ACE/NME caps. Each table
includes the conditioned and any-library success rates, together with mutually
exclusive failure-mode counts such as CA-only, RMSD-only, neither, and—for the
library evaluation—cases where the two metrics pass only against different
conformers.

To create a reconstruction-ready subset containing attempts that pass both
criteria:

```bash
python scripts/experiments/evaluate_pepany_raw_geometry.py \
  --input outputs/cptest_raw \
  --workers 0 \
  --copy-passed-to outputs/cptest_raw_geometry_passed \
  --copy-criterion both
```

`--copy-criterion` accepts `ca`, `10atom`, or `both` for the sampled conditioning
conformer, and `library-ca`, `library-10atom`, or `library-both` for the
any-conformer comparison. Use `library-align-no-clash` to select attempts with
at least one RMSD-passing, clash-free merged conformer. The default is `both`.
This copies each selected candidate PDB/SDF, its conditioning conformer, target,
and raw metadata while preserving the run/target directory layout. For example,
an input at `ThioEther-C/target_1/raw_model_outputs/7.json` is copied below the
same `ThioEther-C/target_1/` path in the new root. A library-based selection also
copies the corresponding best library conformer under
`matching_library_conformers/`. Add `--include-raw-tensors` only when the larger
`.pt` model outputs are also needed.

## Output Layout

For each template, PepAny writes one run directory:

```text
outputs/demo/
  ThioEther-C/
    candidates/
      generation_stats.json
      results.jsonl
      <target_id>/
        pocket.pdb
        10.pdb
        10.sdf
    accepted/
      metadata.csv
      accepted_1.cif
      accepted_2.cif
```

`candidates/` stores raw generated candidates and generation statistics. `accepted/` stores structures that pass the pipeline filters.

The `generation_stats.json` file records total and per-target statistics:

```json
{
  "batches_generated": 10,
  "generated_candidates": 640,
  "generated_candidates_this_run": 640,
  "accepted_total": 25,
  "accept_rate": 0.0391,
  "per_target": {
    "target_id": {
      "generated_candidates": 100,
      "accepted": 5,
      "success_rate": 0.05
    }
  }
}
```

## Resume Generation

Resume mode reads previous accepted structures and `candidates/generation_stats.json`, then computes how many structures remain for each target:

```yaml
resume_generation:
  enabled: true
  previous_dir: outputs/demo
  remaining_sampling_budget: null
```

`previous_dir` can be the same directory as `--save_dir`, as long as it contains the previous template folders with `accepted/` and `candidates/`.

If `remaining_sampling_budget` is set, it overrides `max_sampling_number` for the resumed run:

```yaml
resume_generation:
  enabled: true
  previous_dir: outputs/demo
  remaining_sampling_budget: 20000
```

For example, if the previous run generated 100000 candidates, `remaining_sampling_budget: 20000` makes the effective stopping point 120000 total candidates. This is useful when you want to extend a hard target or cyclization by a controlled number of additional attempts.

## Parallel Generation

Large jobs can be split across GPUs or nodes with `task_split`:

```yaml
task_split:
  enabled: true
  total: 3
  index: 0
```

Equivalent command-line overrides are also available:

```bash
python -m api.generate \
  --config api/demo/config.yaml \
  --ckpt checkpoints/pepany_v1.ckpt \
  --save_dir outputs/demo \
  --gpu 0 \
  --num_shards 3 \
  --shard_id 0
```

Launch one process for each shard, with shard IDs from `0` to `total - 1`. For independent jobs, use different seeds to reduce duplicate sampling:

```bash
python -m api.generate ... --seed 13
python -m api.generate ... --seed 113
python -m api.generate ... --seed 213
```

## Parallel CPU Processing

Candidate post-processing can be parallelized on CPU:

```yaml
generation_postprocess:
  num_workers: 4
  backend: process
  start_method: spawn
  torch_num_threads: 1
```

Post-processing happens after the model samples a batch. It converts sampled tensors into PDB/SDF files, runs template validation, checks ligand amide geometry, and, when the accepted pipeline is enabled, performs cap merging plus RMSD/clash filtering. These steps can be CPU-heavy, so `generation_postprocess.num_workers > 0` can improve throughput when the GPU is waiting for file writing or filtering.

Generation batch prefetch is also configurable:

```yaml
generation_prefetch:
  num_workers: 0
  prefetch_batches: 2
  backend: thread
  allow_process_backend: false
  start_method: spawn
  torch_num_threads: 1
```

Prefetch prepares upcoming generation batches on CPU while the GPU is sampling the current batch. In practice, it calls dataset indexing and collation ahead of time, so it can help when input parsing, pocket construction, or batch assembly becomes a bottleneck. `prefetch_batches` controls how many prepared batches can wait in the queue.

Backend choices:

- `backend: thread` uses worker threads in the same Python process. It has low overhead and shares memory with the main process. This is the default for `generation_prefetch` because it is usually safer for GPU jobs and avoids copying the dataset into separate processes.
- `backend: process` uses separate Python worker processes. It can help for heavier CPU-bound post-processing, which is why it is the default for `generation_postprocess`. For prefetching, process mode is disabled unless `allow_process_backend: true`, because each worker needs its own Python and dataset state and this can be expensive or unstable for large GPU runs.
- `num_workers: 0` disables parallel workers and runs the step in the main process.

`start_method` only applies to `backend: process`. `spawn` starts a clean Python process and is the safest default across macOS/Linux and CUDA-heavy jobs, though it has more startup overhead. `fork` can start faster on Linux, but it inherits process state and is more likely to cause issues with CUDA, RDKit, or already-initialized libraries. Use `forkserver` only if you already know it works well in your environment.

`torch_num_threads` limits PyTorch CPU threads inside each worker. Keeping it at `1` avoids oversubscribing CPU cores when multiple workers are running.

For many GPU workloads, increasing `batch_size` is more beneficial than process-based prefetching. If `max_sampling_number` is set, prefetching is disabled to preserve the exact attempt limit.

## Restore Flattened Relaxed Ligand Chains

If a relaxation/export step has flattened chain `L` into a single `LIG`
residue, restore its `CAP` and amino-acid identities with:

```bash
python scripts/restore_relaxed_l_chain.py \
  --relaxed-dir /path/to/relaxed \
  --metadata-dir /path/to/pepany_cptest_amide_library_accepted/metadata \
  --workers 16 \
  -o /path/to/restored
```

The script recursively matches `*_relaxed.pdb` files to metadata by
cyclization type, target subfolder, and sample number. It automatically
supports PepAny raw PDBs, CPcomposer raw PDBs whose metadata records two
removed terminal placeholder residues, and BoltzGen raw `.cif`/`.cif.gz`
files. For BoltzGen it reads `evaluation.peptide_chain` and applies the
condition-specific core-residue rule used during cap assembly. If recorded
absolute paths have moved, point to their new roots:

```bash
python scripts/restore_relaxed_l_chain.py \
  --relaxed-dir /path/to/relaxed \
  --metadata-dir /path/to/accepted/metadata \
  --raw-dir /path/to/raw_outputs \
  --merged-dir /path/to/accepted/complexes \
  --accepted-output-dir /path/to/accepted_restored
```

Outputs go to `<relaxed-dir>_restored/` by default, together with
`restoration_report.json`. With `-o`, the complete directory tree relative to
`--relaxed-dir` is preserved below the selected output root. When
`--accepted-output-dir` is provided, the unrelaxed accepted PDB is restored
with the exact same peptide/CAP heavy-atom identities and written to the
second output tree, allowing direct name-based pre/post-relaxation RMSD
comparisons. `--workers 0`
automatically uses up to eight CPU processes, `--workers N` selects an explicit
process count, and `--workers 1` runs serially. Coordinates are not changed.
Atom records are grouped by residue, atom serials and `CONECT` references are
updated together, and any missing ligand connectivity is copied from the
unrelaxed merged structure. Use `--pattern '*.pdb'` when relaxed files do not
end in `_relaxed.pdb`. Metadata loading, source indexing, job preparation, and
restoration have progress indicators; rich progress bars use `tqdm` when it is
installed, with periodic text progress as the fallback.
