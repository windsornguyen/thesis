import os
import torch
import torch.distributed as dist

def setup(ddp=False) -> tuple[torch.device, int, int, bool]:
    """
    Adapts to distributed or non-distributed training environments.
    Chooses appropriate backend and device based on the available hardware and environment setup.
    Manages NCCL for NVIDIA GPUs and Gloo for CPUs.
    Returns the device, rank, world size, and a boolean indicating if the current process is the master process.
    """
    if ddp:
        assert torch.cuda.is_available(), "CUDA is required for DDP"
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device, local_rank, rank, world_size, master_process

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
