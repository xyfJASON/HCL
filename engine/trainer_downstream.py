import torch
from engine.trainer import Trainer
from utils.data import get_dataset, get_dataloader


class DownstreamTrainer(Trainer):
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
