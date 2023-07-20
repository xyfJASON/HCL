import os
import tqdm
from argparse import Namespace
from yacs.config import CfgNode as CN

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity

from metrics import BinaryCrossEntropy, IntersectionOverUnion
from models import InpaintNet, RefineNet, Classifier
from utils.logger import get_logger
from utils.mask_blind import DatasetWithMaskBlind
from utils.misc import init_seeds, get_bare_model
from utils.data import get_dataset, get_dataloader
from utils.dist import get_rank, get_world_size, get_local_rank, init_distributed_mode
from utils.dist import main_process_only, is_dist_avail_and_initialized, is_main_process


class Tester:
    def __init__(self, args: Namespace, cfg: CN):
        self.args, self.cfg = args, cfg

        # INITIALIZE DISTRIBUTED MODE
        init_distributed_mode()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # INITIALIZE SEEDS
        init_seeds(self.cfg.seed + get_rank())

        # INITIALIZE LOGGER
        self.logger = get_logger()
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f"Number of devices: {get_world_size()}")

        # BUILD DATASET & DATALOADER
        self.test_set = self.create_data()
        self.logger.info(f'Size of test set: {len(self.test_set)}')

        # BUILD MODELS
        self.inpaintnet = InpaintNet(
            img_size=self.cfg.data.img_size,
            dim=self.cfg.inpaintnet.dim,
            n_conv_stages=self.cfg.inpaintnet.n_conv_stages,
            dim_mults=self.cfg.inpaintnet.dim_mults,
            proj_dim=self.cfg.inpaintnet.proj_dim,
            fuse=self.cfg.inpaintnet.fuse,
            encoder_depths=self.cfg.inpaintnet.encoder_depths,
            decoder_depths=self.cfg.inpaintnet.decoder_depths,
            window_size=self.cfg.inpaintnet.window_size,
            bottleneck_window_size=self.cfg.inpaintnet.bottleneck_window_size,
            bottleneck_depth=self.cfg.inpaintnet.bottleneck_depth,
            conf_threshs=self.cfg.inpaintnet.conf_threshs,
            temperature=self.cfg.inpaintnet.temperature,
            kmeans_n_iters=self.cfg.inpaintnet.kmeans_n_iters,
            kmeans_repeat=self.cfg.inpaintnet.kmeans_repeat,
            gt_mask_to_decoder=False,
            dual_inpainting=False,
            legacy_v=self.cfg.inpaintnet.get('legacy_v', 4),
        )
        self.inpaintnet.to(device=self.device)
        self.inpaintnet.eval()

        self.refinenet = RefineNet(
            dim=self.cfg.refinenet.dim,
            dim_mults=self.cfg.refinenet.dim_mults,
            legacy_v=self.cfg.refinenet.get('legacy_v', 4),
        )
        self.refinenet.to(device=self.device)
        self.refinenet.eval()

        self.classifier = Classifier(dim=self.cfg.inpaintnet.proj_dim)
        self.classifier.to(device=self.device)
        self.classifier.eval()

        self.n_stages = len(self.cfg.inpaintnet.dim_mults)

        # LOAD PRETRAINED WEIGHTS
        if self.cfg.test.pretrained is not None:
            self.load_pretrained(self.cfg.test.pretrained)

        # DISTRIBUTED MODELS
        if is_dist_avail_and_initialized():
            self.inpaintnet = DDP(
                self.inpaintnet,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=True,
            )
            self.refinenet = DDP(
                self.refinenet,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=True,
            )
            self.classifier = DDP(
                self.classifier,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
            )

    def create_data(self):
        test_set = get_dataset(
            name=self.cfg.data.name,
            dataroot=self.cfg.data.dataroot,
            img_size=self.cfg.data.img_size,
            split='test',
        )
        real_dataset_test = None
        if self.cfg.mask.noise_type == 'real':
            real_dataset_test = ConcatDataset([
                get_dataset(
                    name=d['name'],
                    dataroot=d['dataroot'],
                    img_size=d['img_size'],
                    split='test',
                )
                for d in self.cfg.mask.real_dataset
            ])
        test_set = DatasetWithMaskBlind(
            dataset=test_set,
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
            real_dataset=real_dataset_test,
            smooth_radius=self.cfg.mask.smooth_radius,
            is_train=False,
        )
        return test_set

    def load_pretrained(self, model_path: str):
        ckpt = torch.load(model_path, map_location='cpu')
        if ckpt.get('inpaintnet'):
            self.inpaintnet.load_state_dict(ckpt['inpaintnet'])
            self.inpaintnet.to(device=self.device)
            self.logger.info(f'Successfully load inpaintnet from {model_path}')
        else:
            self.logger.warning(f'Fail to load inpaintnet from {model_path}')
        if ckpt.get('refinenet'):
            self.refinenet.load_state_dict(ckpt['refinenet'])
            self.refinenet.to(device=self.device)
            self.logger.info(f'Successfully load refinenet from {model_path}')
        else:
            self.logger.warning(f'Fail to load refinenet from {model_path}')
        if ckpt.get('classifier'):
            self.classifier.load_state_dict(ckpt['classifier'])
            self.classifier.to(device=self.device)
            self.logger.info(f'Successfully load classifier from {model_path}')
        else:
            self.logger.warning(f'Fail to load classifier from {model_path}')

    @torch.no_grad()
    def evaluate(self):
        self.logger.info('Start evaluating...')
        cfg = self.cfg.test

        test_set = self.test_set
        if cfg.n_eval is not None:
            if cfg.n_eval < len(self.test_set):
                test_set = Subset(self.test_set, torch.arange(cfg.n_eval))
                self.logger.info(f"Use a subset of test set, {cfg.n_eval}/{len(self.test_set)}")
            else:
                self.logger.warning(f'Size of test set <= n_eval, ignore n_eval')

        micro_batch = self.cfg.dataloader.micro_batch
        if micro_batch == 0:
            micro_batch = self.cfg.dataloader.batch_size
        self.logger.info(f'Batch size per device: {micro_batch}')
        self.logger.info(f'Effective batch size: {micro_batch * get_world_size()}')
        test_loader = get_dataloader(
            dataset=test_set,
            shuffle=False,
            drop_last=False,
            batch_size=micro_batch,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
            prefetch_factor=self.cfg.dataloader.prefetch_factor,
        )

        metric_mask = [MetricCollection(
            dict(bce=BinaryCrossEntropy(),
                 acc=BinaryAccuracy(multidim_average='samplewise'),
                 f1=BinaryF1Score(multidim_average='samplewise'),
                 iou=IntersectionOverUnion())
        ).to(self.device) for _ in range(self.n_stages)]
        # These metrics expect images to be in [0, 1]
        metric_image = MetricCollection(
            dict(psnr=PeakSignalNoiseRatio(data_range=1, dim=(1, 2, 3)),
                 ssim=StructuralSimilarityIndexMeasure(data_range=1),
                 lpips=LearnedPerceptualImagePatchSimilarity(normalize=True))
        ).to(self.device)
        # FID metric expect images to be in [0, 255] and type uint8
        metric_fid = FrechetInceptionDistance().to(self.device)

        pbar = tqdm.tqdm(test_loader, desc='Evaluating', ncols=120, disable=not is_main_process())
        for X, gt_img, noise, mask in pbar:
            X = X.to(device=self.device, dtype=torch.float32)
            gt_img = gt_img.to(device=self.device, dtype=torch.float32)
            mask = mask.to(device=self.device, dtype=torch.float32)
            recX, projs, conf_mask_hier = self.inpaintnet(X, classifier=self.classifier)
            pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
            refX = self.refinenet(recX, pred_mask)
            for st in range(self.n_stages):
                acc_conf = F.interpolate(conf_mask_hier['acc_confs'][st], size=mask.shape[-2:])
                metric_mask[st].update(acc_conf, mask.long())
            metric_image.update((refX + 1) / 2, (gt_img + 1) / 2)
            metric_fid.update(((refX + 1) / 2 * 255).to(dtype=torch.uint8), real=False)
            metric_fid.update(((gt_img + 1) / 2 * 255).to(dtype=torch.uint8), real=True)
        pbar.close()

        for k, v in metric_image.compute().items():
            self.logger.info(f'{k}: {v.mean()}')
        self.logger.info(f'fid: {metric_fid.compute()}')
        for st in range(self.n_stages):
            for k, v in metric_mask[st].compute().items():
                self.logger.info(f'stage{st}-{k}: {v.mean().item()}')
        self.logger.info('End of evaluation')

    @main_process_only
    @torch.no_grad()
    def sample(self):
        self.logger.info('Start sampling...')
        inpaintnet = get_bare_model(self.inpaintnet)
        refinenet = get_bare_model(self.refinenet)
        classifier = get_bare_model(self.classifier)

        cfg = self.cfg.test
        os.makedirs(cfg.save_dir, exist_ok=True)

        if cfg.random:
            ids = torch.randperm(len(self.test_set))[:cfg.n_samples]
        else:
            ids = torch.arange(cfg.n_samples)

        for i in tqdm.tqdm(ids, desc='Sampling', ncols=120):
            X, gt_img, noise, mask = self.test_set[i]

            X = X.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            recX, projs, conf_mask_hier = inpaintnet(X, classifier=classifier)
            pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
            refX = refinenet(recX, pred_mask)

            gt_img = (gt_img + 1) / 2
            noise = (noise + 1) / 2
            mask = mask.repeat(3, 1, 1).cpu()
            X = (X.squeeze(0).cpu() + 1) / 2
            pred_masks = [
                conf_mask_hier['pred_masks'][0].squeeze(0).repeat(3, 1, 1).cpu(),
                F.interpolate(conf_mask_hier['pred_masks'][1].float(), X.shape[-2:]).squeeze(0).repeat(3, 1, 1).cpu(),
                F.interpolate(conf_mask_hier['pred_masks'][2].float(), X.shape[-2:]).squeeze(0).repeat(3, 1, 1).cpu(),
            ]
            refX = (refX.squeeze(0).cpu() + 1) / 2
            save_image([gt_img, noise, mask, X, pred_masks[2], pred_masks[1], pred_masks[0], refX],
                       os.path.join(cfg.save_dir, str(i.item()) + '.png'),
                       nrow=8, normalize=True, value_range=(0, 1))
        self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
        self.logger.info('End of sampling')
