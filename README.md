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
mkdir checkpoints
wget -c "https://zenodo.org/records/20080102/files/pepany_v1.ckpt?download=1" -O checkpoints/pepany_v1.ckpt
```

Then pass it to generation with `--ckpt`.

## Quick Start

Run peptide generation with the demo config:

```bash
python -m api.generate \
  --config api/demo/config.yaml \
  --ckpt checkpoints/pepany_v1.ckpt \
  --save_dir outputs/demo \
  --gpu 0
```

## Input Config

Generation is controlled by a YAML file. A minimal configuration that generate linear peptide binders looks like:

```yaml
dataset:
  pdb_path: ./api/CPtest/target_with_reference_ligand.pdb
  tgt_chains: R # target chains used as receptor, multiple chains can be specified via the format of [A,B,C]
  lig_chains: L # reference ligand chain used for define pocket, multiple chains can be specified via the format of [A,B,C]
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
    size_from_ligand: true

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
- `max_sampling_number`: maximum number of candidate generations before stopping. Use `null` for no fixed budget (i.e., infinite samping until satisfying the goal_num).
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
