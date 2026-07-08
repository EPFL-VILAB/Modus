#!/bin/bash
#SBATCH --job-name=modus-stage2-16mod
#SBATCH --time=12:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450GB
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/modus_stage2_16mod_%j.out
#SBATCH --error=logs/modus_stage2_16mod_%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --environment=/path/to/your/BAGEL/environment/bagel.toml

# ── Paths ────────────────────────────────────────────────────────────────────
# Every value below is env-overridable (`${VAR:-default}`); the defaults are the
# CSCS 16-node × 4-GPU run. To port to another cluster, `export` the vars before
# `sbatch` — e.g. Apple 8-node × 8-GPU:
#   export MODUS_ROOT=/your/clone num_nodes=8 nproc_per_node=8
# Stage2 resumes from a stage1 output; the default resume_from in
# conf/train/modus_stage2_16mod.yaml is a CSCS absolute path that will NOT exist
# elsewhere. Override it via EXTRA_OVERRIDES (forwarded to train.py as a hydra
# override), e.g.:
#   export EXTRA_OVERRIDES="training.resume_from=/your/stage1_16mod/checkpoints/0014000"
# NOTE: the SBATCH header above (--nodes/--gpus-per-node/--account/--environment)
# is parsed by slurm BEFORE this shell, so it CANNOT read these vars. Keep
# --nodes == num_nodes and --gpus-per-node == nproc_per_node, and override the
# header via `sbatch` CLI flags or by editing it. `--environment` is
# CSCS-pyxis-only; on other clusters drop it and launch in your own container.
MODUS_ROOT="${MODUS_ROOT:-/path/to/your/MODUS}"

# 16-modality stage2 (resumes from a stage1_16mod ckpt; see resume_from in
# conf/train/modus_stage2_16mod.yaml). Validated by smoke_1node.sh.
TRAIN_CONFIG="modus_stage2_16mod"

export output_path="${output_path:-${MODUS_ROOT}/results_paper/stage2/modus_16mod_stage2_from_stage1}"
export ckpt_path="${output_path}/checkpoints"
export num_nodes="${num_nodes:-16}"
export nproc_per_node="${nproc_per_node:-4}"
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# ── Setup ────────────────────────────────────────────────────────────────────
echo "SLURM_NODELIST: $SLURM_NODELIST"

NODES=$(scontrol show hostnames "$SLURM_NODELIST")
NODE_ARRAY=($NODES)
MASTER_NODE="${NODE_ARRAY[0]}"
echo "MASTER_NODE: $MASTER_NODE"

MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname -i | awk '{print $1}')
if [ -z "$MASTER_IP" ]; then
  echo "WARNING: MASTER_IP is empty! Falling back to local hostname -i."
  MASTER_IP=$(hostname -i | awk '{print $1}')
fi
echo "MASTER_IP: $MASTER_IP"

MASTER_PORT=29500
echo "MASTER_PORT: $MASTER_PORT"

export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT="$MASTER_PORT"

mkdir -p "$output_path"
mkdir -p "$ckpt_path"
mkdir -p "${MODUS_ROOT}/logs"

echo "Starting MODUS stage-2 16mod 16-node training..."
echo "MODUS root:           $MODUS_ROOT"
echo "Config:               $TRAIN_CONFIG"
echo "Output directory:     $output_path"
echo "Checkpoint directory: $ckpt_path"

export PYTHONPATH="${MODUS_ROOT}:${PYTHONPATH}"

export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
wandb login "${WANDB_API_KEY:?set WANDB_API_KEY in your environment}"

# NCCL / distributed timeouts
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_TIMEOUT=1800

# ── Launch ───────────────────────────────────────────────────────────────────
srun --mpi=none \
     --cpu-bind=cores \
     --distribution=block:block \
     --export=ALL \
     bash -c '
    cd '"$MODUS_ROOT"'

    NODE_RANK=$SLURM_NODEID
    echo "Running on node rank: $NODE_RANK (host: $(hostname), cwd: $(pwd))"

    torchrun \
      --nnodes='"$num_nodes"' \
      --nproc-per-node='"$nproc_per_node"' \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train.py \
      --config '"$TRAIN_CONFIG"' \
      training.results_dir='"$output_path"' \
      training.checkpoint_dir='"$ckpt_path"' \
      '"$EXTRA_OVERRIDES"'
'
