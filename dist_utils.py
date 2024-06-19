import os
import torch
import torch.distributed as dist
from socket import gethostname

def setup(rank: int, world_size: int, gpus_per_node: int) -> tuple[torch.device, int, int]:
    """
    Adapts to distributed or non-distributed training environments.
    Chooses appropriate backend and device based on the available hardware and environment setup.
    Manages NCCL for NVIDIA GPUs, Gloo for CPUs, and potentially Gloo for Apple Silicon (MPS) in non-distributed setups only.
    Note: Distributed training on MPS is not currently supported.
    """
    local_rank = rank % gpus_per_node if gpus_per_node > 0 else 0
    device = torch.device('cpu')  # Default to CPU
    backend = 'gloo'  # Default backend

    if world_size > 1 and 'SLURM_PROCID' in os.environ:
        if torch.cuda.is_available() and gpus_per_node > 0:
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = 'nccl'
            dist.init_process_group(
                backend=backend, rank=rank, world_size=world_space
            )
            print(f'host: {gethostname()}, rank: {rank}, local_rank: {local_rank}')
            if rank == 0:
                print(f'Group initialized? {dist.is_initialized()}', flush=True)
        elif torch.backends.mps.is_available():
            print(f'MPS detected but distributed training is not supported.')
            # Only setting MPS device without distributed initialization
            device = torch.device('mps')
    else:
        # Non-distributed fallback to the best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')

    return device, local_rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
