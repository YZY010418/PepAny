#!/bin/bash
#SBATCH -J PepAny_CFG_Scan
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o CFGScan_stdout_%A_%a.log
#SBATCH -e CFGScan_stderr_%A_%a.log
#SBATCH --no-requeue
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
source /WORK/PUBLIC/liulei_work/miniconda3/bin/activate
conda activate PepAny

set -Eeuo pipefail

SCALES=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SCALES[@]} )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#SCALES[@]} - 1)); got ${TASK_ID}" >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/fit/liulei/WORK/PepAny}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/api/config1.yaml}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/pepany_v1.ckpt}"
CPCAP_ROOT="${CPCAP_ROOT:-${REPO_ROOT}/api/CPcap}"
CPTEST_DIR="${CPTEST_DIR:-${REPO_ROOT}/api/CPtest}"
CPTEST_ARCHIVE="${CPTEST_ARCHIVE:-${REPO_ROOT}/api/CPtest.tar.gz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/evaluation/CFGScaleScan}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_INDEX="${GPU_INDEX:-0}"
N_TARGETS="${N_TARGETS:-10}"
N_SAMPLES="${N_SAMPLES:-10}"
TARGET_SEED="${TARGET_SEED:-2026}"
GENERATION_SEED="${GENERATION_SEED:-20260724}"
BATCH_SIZE="${BATCH_SIZE:-32}"
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
TARGET_LIST="${TARGET_LIST:-}"
DRY_RUN="${DRY_RUN:-0}"
SCALE="${SCALES[$TASK_ID]}"
SCALE_TAG="${SCALE/./p}"
RUN_DIR="${OUTPUT_ROOT}/scale_${SCALE_TAG}"
RUN_CONFIG="${RUN_DIR}/config.yaml"

cd "$REPO_ROOT"

for required_file in "$BASE_CONFIG" "$CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 2
    fi
done
if [[ ! -d "$CPCAP_ROOT" ]]; then
    echo "CPcap directory not found: ${CPCAP_ROOT}. Set CPCAP_ROOT to the conformer-library directory." >&2
    exit 2
fi
if [[ -n "$TARGET_LIST" && ! -f "$TARGET_LIST" ]]; then
    echo "TARGET_LIST does not exist: ${TARGET_LIST}" >&2
    exit 2
fi
if [[ -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Run directory is not empty: ${RUN_DIR}. Use a new OUTPUT_ROOT to avoid mixing results." >&2
    exit 2
fi

if [[ ! -d "$CPTEST_DIR" ]]; then
    if [[ ! -f "$CPTEST_ARCHIVE" ]]; then
        echo "Neither CPTEST_DIR nor CPTEST_ARCHIVE exists." >&2
        exit 2
    fi
    JOB_TMP="${SLURM_TMPDIR:-${OUTPUT_ROOT}/tmp/${SLURM_JOB_ID:-local}_${TASK_ID}}"
    mkdir -p "$JOB_TMP"
    tar -xzf "$CPTEST_ARCHIVE" -C "$JOB_TMP"
    CPTEST_DIR="${JOB_TMP}/CPtest"
fi

mkdir -p "$RUN_DIR"

"$PYTHON_BIN" - \
    "$BASE_CONFIG" "$RUN_CONFIG" "$CPTEST_DIR" "$CPCAP_ROOT" "$TARGET_LIST" \
    "$N_TARGETS" "$N_SAMPLES" "$TARGET_SEED" "$GENERATION_SEED" "$BATCH_SIZE" "$SCALE" <<'PY'
import random
import sys
from pathlib import Path

import yaml


(
    base_config_path,
    run_config_path,
    cptest_dir,
    cpcap_root,
    target_list_path,
    n_targets,
    n_samples,
    target_seed,
    generation_seed,
    batch_size,
    scale,
) = sys.argv[1:]

n_targets = int(n_targets)
n_samples = int(n_samples)
target_seed = int(target_seed)
generation_seed = int(generation_seed)
batch_size = int(batch_size)
scale = float(scale)
cptest_dir = Path(cptest_dir).resolve()
cpcap_root = Path(cpcap_root).resolve()

with Path(base_config_path).open() as handle:
    config = yaml.safe_load(handle)


def resolve_listed_target(value):
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = cptest_dir / path
    return candidate if candidate.is_file() else Path(base_config_path).resolve().parent / path


if target_list_path:
    with Path(target_list_path).open() as handle:
        listed_targets = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    targets = [resolve_listed_target(value).resolve() for value in listed_targets]
    if len(targets) != n_targets:
        raise ValueError(f"TARGET_LIST must contain exactly {n_targets} entries; found {len(targets)}")
else:
    candidates = sorted(cptest_dir.glob("*.pdb"))
    if len(candidates) < n_targets:
        raise ValueError(f"Need at least {n_targets} PDB files in {cptest_dir}; found {len(candidates)}")
    rng = random.Random(target_seed)
    original_targets = [path for path in candidates if "7cjb" in path.name.lower()]
    selected = original_targets[:1]
    remaining = [path for path in candidates if path not in selected]
    selected.extend(rng.sample(remaining, n_targets - len(selected)))
    targets = sorted(path.resolve() for path in selected)

missing_targets = [str(path) for path in targets if not path.is_file()]
if missing_targets:
    raise FileNotFoundError(f"Target PDB files not found: {missing_targets}")

dataset = config.setdefault("dataset", {})
for key in ("pdb_path", "pdb_paths", "pdb_dir", "pdb_pattern", "recursive"):
    dataset.pop(key, None)
dataset["pdb_paths"] = [str(path) for path in targets]

config["n_samples"] = n_samples
config["max_sampling"] = len(targets) * n_samples
config["seed"] = generation_seed
config["batch_size"] = batch_size
config.setdefault("sample_opt", {})["guidance_scale"] = scale
config["amide_bond_check"] = {"enabled": False}
config["resume_generation"] = {"enabled": False, "previous_dir": None}
config["task_split"] = {"enabled": False, "total": 1, "index": 0}

expected_types = {
    "Alkene-A",
    "Amide-A",
    "Disulfide-A",
    "Ester-A",
    "ThioEther-A",
    "ThioEther-B",
    "ThioEther-C",
    "Triazole-A",
}
templates = config.get("templates", [])
found_types = {template.get("name") for template in templates}
if found_types != expected_types:
    raise ValueError(f"Expected the eight CPSea2 cyclization templates; found {sorted(found_types)}")

for template in templates:
    cap_config = template.get("term_rel_pos_sdf")
    if not cap_config:
        raise ValueError(f"Template {template['name']} has no term_rel_pos_sdf configuration")
    original_name = Path(cap_config["path"]).name
    cap_config["path"] = str(cpcap_root / template["name"] / original_name)
    if not Path(cap_config["path"]).exists():
        raise FileNotFoundError(f"CPcap conformer path not found: {cap_config['path']}")
    template["pipeline"] = {"enabled": False}

run_config_path = Path(run_config_path)
run_config_path.parent.mkdir(parents=True, exist_ok=True)
with run_config_path.open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"CFG scale: {scale}")
print(f"Samples: {len(targets)} targets x {len(templates)} types x {n_samples} = {len(targets) * len(templates) * n_samples}")
print("Selected targets:")
for target in targets:
    print(f"  {target.name}")
PY

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete: ${RUN_CONFIG}"
    exit 0
fi

"$PYTHON_BIN" -m api.generate \
    --config "$RUN_CONFIG" \
    --ckpt "$CHECKPOINT" \
    --save_dir "$RUN_DIR" \
    --gpu "$GPU_INDEX" \
    --seed "$GENERATION_SEED" \
    --postprocess_workers "$POSTPROCESS_WORKERS"

touch "${RUN_DIR}/.complete"
echo "Completed CFG scale ${SCALE}: ${RUN_DIR}"
