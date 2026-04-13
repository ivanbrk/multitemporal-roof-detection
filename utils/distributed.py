import os
import socket

import torch
import torch.distributed as dist


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def init_distributed(rank, world_size, master_port):
    if world_size <= 1:
        return False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return True


def distributed_barrier():
    if not dist.is_available() or not dist.is_initialized():
        return
    device_ids = None
    if str(dist.get_backend()).lower() == "nccl" and torch.cuda.is_available():
        device_ids = [torch.cuda.current_device()]
    dist.barrier(device_ids=device_ids)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_sum_tensor(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
