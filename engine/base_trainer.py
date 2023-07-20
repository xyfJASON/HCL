import os
from argparse import Namespace
from yacs.config import CfgNode as CN

import torch
from torch.utils.data import ConcatDataset

from utils.logger import get_logger, StatusTracker
from utils.mask_blind import DatasetWithMaskBlind
from utils.data import get_dataset, get_dataloader
from utils.misc import get_time_str, init_seeds, create_exp_dir
from utils.dist import get_rank, get_world_size, init_distributed_mode, broadcast_objects


class BaseTrainer:
    def __init__(self, args: Namespace, cfg: CN):
        self.args, self.cfg = args, cfg
        self.time_str = get_time_str()

        # INITIALIZE DISTRIBUTED MODE
        init_distributed_mode()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # INITIALIZE SEEDS
        init_seeds(self.cfg.seed + get_rank())

        # CREATE EXPERIMENT DIRECTORY
        self.exp_dir = create_exp_dir(
            cfg_dump=self.cfg.dump(sort_keys=False),
            resume=self.cfg.train.resume is not None,
            time_str=self.time_str,
            name=self.cfg.train.exp_dir,
            no_interaction=self.args.no_interaction,
        )
        self.exp_dir = broadcast_objects(self.exp_dir)

        # INITIALIZE LOGGER
        self.logger = get_logger(log_file=os.path.join(self.exp_dir, f'output-{self.time_str}.log'))
        self.logger.info(f'Experiment directory: {self.exp_dir}')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f"Number of devices: {get_world_size()}")

        # BUILD DATASET & DATALOADER
        self.micro_batch = self.cfg.dataloader.micro_batch
        if self.micro_batch == 0:
            self.micro_batch = self.cfg.dataloader.batch_size
        self.train_set, self.valid_set, self.train_loader, self.valid_loader = self.create_data()
        effective_batch = self.cfg.dataloader.batch_size * get_world_size()
        self.logger.info(f'Size of training set: {len(self.train_set)}')
        self.logger.info(f'Batch size per device: {self.cfg.dataloader.batch_size}')
        self.logger.info(f'Effective batch size: {effective_batch}')

        # DEFINE STATUS TRACKER
        self.status_tracker = StatusTracker(
            logger=self.logger,
            exp_dir=self.exp_dir,
            print_freq=self.cfg.train.print_freq,
        )

        # BUILD MODELS AND OPTIMIZERS
        # LOAD PRETRAINED WEIGHTS
        # RESUME
        # DEFINE LOSSES
        # DISTRIBUTED MODELS
        # EVALUATION METRICS

    def create_data(self):
        train_set = get_dataset(
            name=self.cfg.data.name,
            dataroot=self.cfg.data.dataroot,
            img_size=self.cfg.data.img_size,
            split='train',
        )
        valid_set = get_dataset(
            name=self.cfg.data.name,
            dataroot=self.cfg.data.dataroot,
            img_size=self.cfg.data.img_size,
            split='valid',
            subset_ids=torch.arange(3000),
        )
        real_dataset_train = None
        real_dataset_valid = None
        if self.cfg.mask.noise_type == 'real':
            real_dataset_train = ConcatDataset([
                get_dataset(
                    name=d['name'],
                    dataroot=d['dataroot'],
                    img_size=d['img_size'],
                    split='train',
                ) for d in self.cfg.mask.real_dataset
            ])
            real_dataset_valid = ConcatDataset([
                get_dataset(
                    name=d['name'],
                    dataroot=d['dataroot'],
                    img_size=d['img_size'],
                    split='valid',
                )
                for d in self.cfg.mask.real_dataset
            ])
        train_set = DatasetWithMaskBlind(
            dataset=train_set,
            mask_type=self.cfg.mask.mask_type,
            dir_path=getattr(self.cfg.mask, 'dir_path', None),
            dir_invert_color=getattr(self.cfg.mask, 'dir_invert_color', False),
            rect_num=getattr(self.cfg.mask, 'rect_num', (0, 4)),
            rect_length_ratio=getattr(self.cfg.mask, 'rect_length_ratio', (0.2, 0.8)),
            brush_num=getattr(self.cfg.mask, 'brush_num', (1, 9)),
            brush_turns=getattr(self.cfg.mask, 'brush_turns', (4, 18)),
            brush_width_ratio=getattr(self.cfg.mask, 'brush_width_ratio', (0.02, 0.1)),
            brush_length_ratio=getattr(self.cfg.mask, 'brush_length_ratio', (0.1, 0.25)),
            noise_type=getattr(self.cfg.mask, 'noise_type', 'constant'),
            constant_value=getattr(self.cfg.mask, 'constant_value', (0, 0, 0)),
            real_dataset=real_dataset_train,
            smooth_radius=self.cfg.mask.smooth_radius,
            is_train=True,
        )
        valid_set = DatasetWithMaskBlind(
            dataset=valid_set,
            mask_type=self.cfg.mask.mask_type,
            dir_path=getattr(self.cfg.mask, 'dir_path', None),
            dir_invert_color=getattr(self.cfg.mask, 'dir_invert_color', False),
            rect_num=getattr(self.cfg.mask, 'rect_num', (0, 4)),
            rect_length_ratio=getattr(self.cfg.mask, 'rect_length_ratio', (0.2, 0.8)),
            brush_num=getattr(self.cfg.mask, 'brush_num', (1, 9)),
            brush_turns=getattr(self.cfg.mask, 'brush_turns', (4, 18)),
            brush_width_ratio=getattr(self.cfg.mask, 'brush_width_ratio', (0.02, 0.1)),
            brush_length_ratio=getattr(self.cfg.mask, 'brush_length_ratio', (0.1, 0.25)),
            noise_type=getattr(self.cfg.mask, 'noise_type', 'constant'),
            constant_value=getattr(self.cfg.mask, 'constant_value', (0, 0, 0)),
            real_dataset=real_dataset_valid,
            smooth_radius=self.cfg.mask.smooth_radius,
            is_train=False,
        )
        train_loader = get_dataloader(
            dataset=train_set,
            shuffle=True,
            drop_last=True,
            batch_size=self.cfg.dataloader.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
            prefetch_factor=self.cfg.dataloader.prefetch_factor,
        )
        valid_loader = get_dataloader(
            dataset=valid_set,
            shuffle=False,
            drop_last=False,
            batch_size=self.micro_batch,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
            prefetch_factor=self.cfg.dataloader.prefetch_factor,
        )
        return train_set, valid_set, train_loader, valid_loader

    def run_loop(self):
        raise NotImplementedError

    def save_ckpt(self, save_path):
        raise NotImplementedError

    def load_ckpt(self, ckpt_path):
        raise NotImplementedError
