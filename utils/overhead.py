"""
Calculate #params, FLOPs/MACs, inference time for a given model.
"""
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

import fvcore.nn


def count_params(model: nn.Module, recursive: bool = False, max_depth: int = 3):
    """
    Args:
        model (nn.Module): pytorch model
        recursive (bool): recursively count parameters and print them in a table
        max_depth (int): maximum depth to recursively print submodules or parameters
    """
    print(f'Total number of parameters: {sum([p.numel() for p in model.parameters()])}')
    print(f'Total number of learnable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
    if recursive:
        print(fvcore.nn.parameter_count_table(model, max_depth))


def calc_flops(model: nn.Module,
               dummy_input: Union[Tensor, Tuple[Tensor, ...]],
               recursive: bool = False,
               max_depth: int = 3):
    flops = fvcore.nn.FlopCountAnalysis(model, dummy_input)
    print('FLOPs (actually MACs):')
    if recursive:
        print(fvcore.nn.flop_count_table(flops, max_depth=max_depth))
    else:
        print(flops.total())


@torch.no_grad()
def calc_inference_time(model: nn.Module,
                        dummy_input: Union[Tensor, Tuple[Tensor, ...]],
                        repetitions: int = 300):
    """
    References: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    """
    model.eval()
    if not torch.cuda.is_available():
        print('cuda is not available')
        return
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    # GPU warm-up
    if not isinstance(dummy_input, (list, tuple)):
        dummy_input = (dummy_input, )
    for _ in range(10):
        _ = model(*dummy_input)
    # measure performance
    for rep in range(repetitions):
        starter.record()
        _ = model(*dummy_input)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
    print(f'Inference time: {sum(timings) / len(timings)}')
