import os
import sys
import random
import shutil
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dist import main_process_only


def init_seeds(seed: int = 0, cuda_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def check_freq(freq: int, step: int):
    assert isinstance(freq, int)
    return freq >= 1 and (step + 1) % freq == 0


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def image_float_to_uint8(image: torch.Tensor):
    assert image.dtype == torch.float32
    assert (0 <= image).all() and (image <= 1).all()
    return (image * 255).to(dtype=torch.uint8)


def image_norm_to_float(image: torch.Tensor):
    assert image.dtype == torch.float32
    assert (-1 <= image).all() and (image <= 1).all()
    return (image + 1) / 2


def image_norm_to_uint8(image: torch.Tensor):
    assert image.dtype == torch.float32
    assert (-1 <= image).all() and (image <= 1).all()
    return ((image + 1) / 2 * 255).to(dtype=torch.uint8)


def get_bare_model(model: nn.Module or DDP):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def find_resume_checkpoint(exp_dir: str, resume: str):
    """ Checkpoints are named after 'stepxxxxxx.pt' """
    if os.path.isfile(resume):
        ckpt_path = resume
    elif resume == 'best':
        ckpt_path = os.path.join(exp_dir, 'ckpt', 'best.pt')
    elif resume == 'latest':
        d = dict()
        for filename in os.listdir(os.path.join(exp_dir, 'ckpt')):
            name, ext = os.path.splitext(filename)
            if ext == '.pt' and name[:4] == 'step':
                d.update({int(name[4:]): filename})
        ckpt_path = os.path.join(exp_dir, 'ckpt', d[sorted(d)[-1]])
    else:
        raise ValueError(f'resume option {resume} is invalid')
    assert os.path.isfile(ckpt_path), f'{ckpt_path} is not a .pt file'
    return ckpt_path


@main_process_only
def create_exp_dir(cfg_dump: str, resume: bool = False, time_str: str = None,
                   name: str = None, no_interaction: bool = False):
    if time_str is None:
        time_str = get_time_str()
    if name is None:
        name = f'exp-{time_str}'
    exp_dir = os.path.join('runs', name)
    if os.path.exists(exp_dir) and not resume:
        cover = True
        if not no_interaction:
            cover = query_yes_no(
                question=f'{exp_dir} already exists! Cover it anyway?',
                default='no',
            )
        if cover:
            shutil.rmtree(exp_dir, ignore_errors=True)
        else:
            sys.exit(1)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    with open(os.path.join(exp_dir, f'config-{time_str}.yaml'), 'w') as f:
        f.write(cfg_dump)
    return exp_dir


def query_yes_no(question: str, default: str = "yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Copied from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
