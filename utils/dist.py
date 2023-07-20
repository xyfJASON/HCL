import os
import functools

import torch
import torch.distributed as dist


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print(f"Distributed init (rank {int(os.environ['RANK'])})", flush=True)
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')
        dist.barrier()
    else:
        print('Not using distributed mode')


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def sync_params(params):
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def all_reduce_mean(tensor):
    world_size = get_world_size()
    if world_size > 1:
        rt = tensor.clone() / world_size
        dist.all_reduce(rt)
        return rt
    else:
        return tensor


def broadcast_objects(objects):
    if is_dist_avail_and_initialized():
        islist = isinstance(objects, list)
        if not islist:
            objects = [objects]
        dist.broadcast_object_list(objects, src=0)
        if not islist:
            objects = objects[0]
    return objects
