#!/bin/bash
#SBATCH --job-name=ssm-pt              # Name of the job
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gres=gpu:4                     # Total number of GPUs requested
#SBATCH --ntasks-per-node=4              # Total number of tasks per node
#SBATCH --cpus-per-task=1                # CPU-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=320G                        # Memory per node (4G is default)
#SBATCH --time=4:00:00                   # Job time limit
## SBATCH --partition=pli                  # Specify the node partition
## SBATCH --account=spectralssmtorch       # Project name


# Check if SLURM environment variables are available and exit if not
if [[ -z $SLURM_NNODES ]] || [[ -z $SLURM_NTASKS_PER_NODE ]]; then
    echo "SLURM environment variables not set. Assuming standalone setup."
    WORLD_SIZE=1
else
    # Calculate WORLD_SIZE safely
    WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
    echo "WORLD_SIZE=$WORLD_SIZE"
fi

# Determine the master node address
if [[ -z $SLURM_JOB_NODELIST ]]; then
    echo "SLURM_JOB_NODELIST not set. Using localhost as DIST_MASTER_ADDR."
    DIST_MASTER_ADDR="localhost"
else
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    if [[ -z $master_addr ]]; then
        echo "Failed to determine DIST_MASTER_ADDR. Exiting."
        exit 1
    fi
    DIST_MASTER_ADDR=$master_addr
    echo "DIST_MASTER_ADDR=$DIST_MASTER_ADDR"
fi

# Set an available port for the DIST_MASTER_PORT
if [[ -z $SLURM_JOBID ]]; then
    export DIST_MASTER_PORT=$(shuf -i 2000-65000 -n 1)
else
    export DIST_MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi
echo "DIST_MASTER_PORT=$DIST_MASTER_PORT"

# Optional WandB environment settings, uncomment to use
# export WANDB_ENTITY=windsornguyen
# export WANDB_API_KEY=17dce35b188763800b6e9a443a761a1e713d87ab
# export WANDB_PROJECT=compass-finetune
# export WANDB_LOG_MODEL=checkpoint
export WANDB_DISABLED=True

echo "WandB and Slurm Environment Variables:"
printenv | grep -E 'WANDB|SLURM' | sort

# API key check
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY: Not found"
else
    echo "WANDB_API_KEY: Found"
fi

# Display the hostname
echo "Running on host $(hostname)"

# Print the GPU information, check for CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=gpu_name --format=csv,noheader
else
    echo "CUDA not installed or GPUs not available."
fi

# Environment module management
# module purge
# module load anaconda3/2024.2
# module load cudatoolkit/12.4

# Activate the conda environment
# conda activate thesis
# if [ $? -ne 0 ]; then
#     echo "Failed to activate conda environment. Exiting."
#     exit 1
# fi

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Check if the test.py script exists
if [ ! -f "test.py" ]; then
    echo "test.py script not found. Exiting."
    exit 1
fi

# Determine the execution mode based on WORLD_SIZE
if [[ "$WORLD_SIZE" -eq "1" ]]; then
    torchrun --standalone --rdzv_backend=c10d --rdzv_endpoint=$DIST_MASTER_ADDR:$DIST_MASTER_PORT pretrain.py --model stu
else
    torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE --rdzv_backend=c10d --rdzv_endpoint=$DIST_MASTER_ADDR:$DIST_MASTER_PORT pretrain.py --model stu
fi

##################### HELPFUL DELLA COMMANDS #####################

# Disk usage and quota info; request additional space via link.
# `checkquota`

# Operating system details.
# `cat /etc/os-release`

# CPU specifications on the current node.
# `lscpu`

# Compute node information (standard and easy-to-read formats).
# `snodes`
# `shownodes`

# Cluster nodes usage overview; check status (idle, down, busy).
# `sinfo`
    # GPU usage specifics.
    # `sinfo -p gpu`

# Quality of Service insights: job partitions and limits.
# `qos`

# Current processor activity; exit with 'q'.
# `top`
# `htop`

# Overview of group-assigned cluster shares.
# `sshare`

# Job priority mechanics: factors and weights.
# `sprio -w`

# Performance of a completed job by job ID.
# `seff <jobid>`

# Your historical job records.
# `shistory`

# Detailed job statistics (memory, CPU, GPU).
# `jobstats <jobid>`

# Additional commands from your list for GPU details:

# Details about GPUs on the cluster.
# `snodes`

# Number of available GPUs.
# `shownodes -p gpu,mig`

# GPU utilization, refreshed every 10 min.
# `gpudash`
    # Specific to your jobs.
    # `gpudash -u $USER`

 # Real-time GPU status on active jobs.
# `nvidia-smi` OR `watch nvidia-smi`

 # Your queue status.
# `squeue -u $USER`