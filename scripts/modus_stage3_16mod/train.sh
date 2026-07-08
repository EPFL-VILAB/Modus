#!/bin/bash
#SBATCH --job-name=modus-stage3-16mod
#SBATCH --time=12:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450GB
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/modus_stage3_16mod_%j.out
#SBATCH --error=logs/modus_stage3_16mod_%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --environment=/path/to/your/BAGEL/environment/bagel.toml

MODUS_ROOT="${MODUS_ROOT:-/path/to/your/MODUS}"
TRAIN_CONFIG="modus_stage3_16mod"

STAGE2_CKPT="${MODUS_ROOT}/results_paper/stage2/modus_16mod_stage2_from_stage1/checkpoints/0020000"
STAGE2_MODEL_SHARD="${STAGE2_CKPT}/model.00000-of-00008.safetensors"

export output_path="${MODUS_ROOT}/results_paper/stage3/modus_16mod_stage3_from_stage2"
export ckpt_path="${output_path}/checkpoints"
export num_nodes=16
export nproc_per_node=4

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

echo "Starting MODUS stage-3 16mod 16-node training..."
echo "MODUS root:           $MODUS_ROOT"
echo "Config:               $TRAIN_CONFIG"
echo "Stage 2 checkpoint:   $STAGE2_CKPT"
echo "Output directory:     $output_path"
echo "Checkpoint directory: $ckpt_path"

# The job is normally submitted after the stage2 job, but checkpoint saving can
# lag job state on shared storage. Wait briefly rather than failing immediately.
# Training checkpoints are saved as FSDP shards; train/fsdp_utils.py loads these
# directly, so do not require a merged model.safetensors here.
for i in $(seq 1 120); do
  if [ -f "$STAGE2_MODEL_SHARD" ]; then
    break
  fi
  echo "Waiting for $STAGE2_MODEL_SHARD ($i/120)"
  sleep 60
done

if [ ! -f "$STAGE2_MODEL_SHARD" ]; then
  echo "ERROR: missing stage2 20k model shard: $STAGE2_MODEL_SHARD"
  exit 1
fi

export PYTHONPATH="${MODUS_ROOT}:${PYTHONPATH}"

export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
wandb login "${WANDB_API_KEY:?set WANDB_API_KEY in your environment}"

export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_TIMEOUT=1800

# Keep the stage2 fix: do not force all 13 groups into every packed batch.
export HUNYUAN_DISABLE_MANDATORY_WARMSTART=${HUNYUAN_DISABLE_MANDATORY_WARMSTART:-1}
export HUNYUAN_BATCH_SHAPE_DEBUG=${HUNYUAN_BATCH_SHAPE_DEBUG:-0}
export HUNYUAN_BATCH_SHAPE_DEBUG_STEPS=${HUNYUAN_BATCH_SHAPE_DEBUG_STEPS:-80}
export HUNYUAN_STAGE_TIMING_DEBUG=${HUNYUAN_STAGE_TIMING_DEBUG:-0}
export HUNYUAN_STAGE_TIMING_DEBUG_STEPS=${HUNYUAN_STAGE_TIMING_DEBUG_STEPS:-80}

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
      training.resume_from='"$STAGE2_CKPT"' \
      training.results_dir='"$output_path"' \
      training.checkpoint_dir='"$ckpt_path"' \
      training.save_every=200 \
      training.wandb_name="modus_16mod_stage3_from_stage2"
'
