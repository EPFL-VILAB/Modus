#!/bin/bash
#SBATCH --job-name=modus-stage2-16mod-smoke
#SBATCH --time=0:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450GB
#SBATCH --gpus-per-node=4
#SBATCH --partition=debug
#SBATCH --output=logs/modus_stage2_16mod_smoke_%j.out
#SBATCH --error=logs/modus_stage2_16mod_smoke_%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --environment=/path/to/your/BAGEL/environment/bagel.toml

# 1-node smoke test for stage2 16mod. Verifies:
#   - stage1 ckpt loads (no missing/unexpected keys)
#   - step 0 loss is finite + sane (~4.4-4.5 like stage1 14k end)
#   - data pipeline serves all 12 unified_any2X targets
#   - no NaN through ~50 steps
# DOES NOT write checkpoints (save_every overridden to 100000).

MODUS_ROOT="${MODUS_ROOT:-/path/to/your/MODUS}"
TRAIN_CONFIG="modus_stage2_16mod"
STAGE1_CKPT="${MODUS_ROOT}/results_paper/stage1/modus_16mod_stage1_cocodet/checkpoints/0014000"

export output_path="${MODUS_ROOT}/results_paper/stage2/modus_16mod_stage2_smoke"
export ckpt_path="${output_path}/checkpoints"
export num_nodes=1
export nproc_per_node=4

echo "SLURM_NODELIST: $SLURM_NODELIST"

NODES=$(scontrol show hostnames "$SLURM_NODELIST")
NODE_ARRAY=($NODES)
MASTER_NODE="${NODE_ARRAY[0]}"
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname -i | awk '{print $1}')
[ -z "$MASTER_IP" ] && MASTER_IP=$(hostname -i | awk '{print $1}')
MASTER_PORT=29501

export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT="$MASTER_PORT"

mkdir -p "$output_path" "$ckpt_path" "${MODUS_ROOT}/logs"

echo "Starting MODUS stage-2 16mod SMOKE test (1 node, 50 steps)..."
echo "Stage 1 ckpt: $STAGE1_CKPT"
echo "Output:       $output_path"

export PYTHONPATH="${MODUS_ROOT}:${PYTHONPATH}"
export WANDB_MODE=disabled
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_TIMEOUT=1800
export HUNYUAN_DISABLE_MANDATORY_WARMSTART=${HUNYUAN_DISABLE_MANDATORY_WARMSTART:-1}
export HUNYUAN_BATCH_SHAPE_DEBUG=${HUNYUAN_BATCH_SHAPE_DEBUG:-1}
export HUNYUAN_BATCH_SHAPE_DEBUG_STEPS=${HUNYUAN_BATCH_SHAPE_DEBUG_STEPS:-80}
export HUNYUAN_STAGE_TIMING_DEBUG=${HUNYUAN_STAGE_TIMING_DEBUG:-1}
export HUNYUAN_STAGE_TIMING_DEBUG_STEPS=${HUNYUAN_STAGE_TIMING_DEBUG_STEPS:-80}

# Stage1 ckpt was saved with num_shard=8; smoke runs num_shard=4.
# Force-load from consolidated model.safetensors instead of per-shard files.
export MODUS_FORCE_FULL_STATE_LOAD=1

srun --mpi=none \
     --cpu-bind=cores \
     --distribution=block:block \
     --export=ALL \
     bash -c '
    cd '"$MODUS_ROOT"'

    NODE_RANK=$SLURM_NODEID
    echo "Smoke node rank: $NODE_RANK (host: $(hostname))"

    torchrun \
      --nnodes='"$num_nodes"' \
      --nproc-per-node='"$nproc_per_node"' \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train.py \
      --config '"$TRAIN_CONFIG"' \
      training.resume_from='"$STAGE1_CKPT"' \
      training.results_dir='"$output_path"' \
      training.checkpoint_dir='"$ckpt_path"' \
      training.total_steps=50 \
      training.save_every=100000 \
      training.log_every=1 \
      training.num_shard=4 \
      training.num_replicate=1 \
      training.wandb_name="modus_16mod_stage2_smoke"
'
