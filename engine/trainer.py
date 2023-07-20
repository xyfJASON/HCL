import os
import tqdm
from argparse import Namespace
from contextlib import nullcontext
from yacs.config import CfgNode as CN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from metrics import KeyValueAverageMeter
from metrics.bce import BinaryCrossEntropy

from engine.base_trainer import BaseTrainer
from losses import HierarchicalContrastiveLoss, PerceptualLoss, AdversarialLoss
from models import InpaintNet, RefineNet, PatchDiscriminator, VGG19FeatureExtractor, Classifier
from utils.data import get_data_generator
from utils.optimizer import optimizer_to_device
from utils.misc import check_freq, get_bare_model, find_resume_checkpoint
from utils.dist import main_process_only, get_local_rank, is_dist_avail_and_initialized, is_main_process


class Trainer(BaseTrainer):
    def __init__(self, args: Namespace, cfg: CN):
        super().__init__(args, cfg)

        # BUILD MODELS AND OPTIMIZERS
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
            gt_mask_to_decoder=True if self.args.phase == 1 else False,
            dual_inpainting=False,
        )
        self.inpaintnet.to(device=self.device)
        self.optimizer_inpaintnet = optim.Adam(
            params=self.inpaintnet.parameters(),
            lr=self.cfg.train.lr_inpaintnet,
        )

        self.refinenet = RefineNet(
            dim=self.cfg.refinenet.dim,
            dim_mults=self.cfg.refinenet.dim_mults,
        )
        self.refinenet.to(device=self.device)
        self.optimizer_refinenet = optim.Adam(
            params=self.refinenet.parameters(),
            lr=self.cfg.train.lr_refinenet,
        )

        self.classifier = Classifier(dim=self.cfg.inpaintnet.proj_dim)
        self.classifier.to(device=self.device)
        self.optimizer_classifier = optim.Adam(
            params=self.classifier.parameters(),
            lr=self.cfg.train.lr_inpaintnet,
        )

        self.pdisc = PatchDiscriminator()
        self.pdisc.to(device=self.device)
        self.optimizer_pdisc = optim.Adam(
            params=self.pdisc.parameters(),
            lr=self.cfg.train.lr_pdisc,
        )

        vgg = VGG19FeatureExtractor()
        vgg.to(device=self.device)

        # LOAD PRETRAINED WEIGHTS
        if self.cfg.train.pretrained is not None:
            self.load_pretrained(self.cfg.train.pretrained)

        # RESUME
        self.cur_step = 0
        if self.cfg.train.resume is not None:
            resume_path = find_resume_checkpoint(self.exp_dir, self.cfg.train.resume)
            self.logger.info(f'Resume from {resume_path}')
            self.load_ckpt(resume_path)

        # DEFINE LOSSES
        self.hier_contrast = HierarchicalContrastiveLoss(
            start_stage=self.cfg.contrast.get('start_stage', 0),
            total_stages=len(self.cfg.inpaintnet.dim_mults),
            temperature=self.cfg.contrast.temperature,
            sample_num=self.cfg.contrast.sample_num,
            valid_thresh=self.cfg.contrast.valid_thresh,
            invalid_thresh=self.cfg.contrast.invalid_thresh,
            hard_mining=self.cfg.contrast.hard_mining,
            hard_num=self.cfg.contrast.hard_num,
        )
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(
            feature_extractor=vgg,
            use_features=self.cfg.train.perc_use_features,
        )
        self.perceptual_refined = PerceptualLoss(
            feature_extractor=vgg,
            use_features=self.cfg.train.perc_use_features_refined,
        )
        self.patch_adv = AdversarialLoss(self.pdisc)
        self.bcewithlogits = nn.BCEWithLogitsLoss()

        self.loss_meter_gen = KeyValueAverageMeter(
            keys=['loss_rec', 'loss_rec_refined', 'loss_perc', 'loss_perc_refined',
                  'loss_adv_refined', 'loss_contrast', 'loss_cls']
        ).to(self.device)
        self.loss_meter_pdisc = KeyValueAverageMeter(keys=['lossD']).to(self.device)

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
            self.pdisc = DDP(
                self.pdisc,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
            )

        # EVALUATION METRICS
        self.metric_image = MetricCollection(
            dict(psnr=PeakSignalNoiseRatio(data_range=1., dim=(1, 2, 3)),
                 ssim=StructuralSimilarityIndexMeasure(data_range=1.))
        ).to(self.device)
        self.metric_mask = MetricCollection(
            dict(bce=BinaryCrossEntropy(),
                 acc=BinaryAccuracy(multidim_average='samplewise'),
                 f1=BinaryF1Score(multidim_average='samplewise'))
        ).to(self.device)

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
        if ckpt.get('pdisc'):
            self.pdisc.load_state_dict(ckpt['pdisc'])
            self.pdisc.to(device=self.device)
            self.logger.info(f'Successfully load pdisc from {model_path}')
        else:
            self.logger.warning(f'Fail to load pdisc from {model_path}')

    def load_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.load_pretrained(ckpt_path)
        self.optimizer_inpaintnet.load_state_dict(ckpt['optimizer_inpaintnet'])
        self.optimizer_refinenet.load_state_dict(ckpt['optimizer_refinenet'])
        self.optimizer_classifier.load_state_dict(ckpt['optimizer_classifier'])
        self.optimizer_pdisc.load_state_dict(ckpt['optimizer_pdisc'])
        optimizer_to_device(self.optimizer_inpaintnet, self.device)
        optimizer_to_device(self.optimizer_refinenet, self.device)
        optimizer_to_device(self.optimizer_classifier, self.device)
        optimizer_to_device(self.optimizer_pdisc, self.device)
        self.cur_step = ckpt['step'] + 1

    @main_process_only
    def save_ckpt(self, save_path: str):
        state_dicts = dict(
            inpaintnet=get_bare_model(self.inpaintnet).my_state_dict(),
            refinenet=get_bare_model(self.refinenet).state_dict(),
            classifier=get_bare_model(self.classifier).state_dict(),
            pdisc=get_bare_model(self.pdisc).state_dict(),
            optimizer_inpaintnet=self.optimizer_inpaintnet.state_dict(),
            optimizer_refinenet=self.optimizer_refinenet.state_dict(),
            optimizer_classifier=self.optimizer_classifier.state_dict(),
            optimizer_pdisc=self.optimizer_pdisc.state_dict(),
            step=self.cur_step,
        )
        torch.save(state_dicts, save_path)

    def run_loop(self):
        self.logger.info('Start training...')
        train_data_generator = get_data_generator(
            dataloader=self.train_loader,
            start_epoch=self.cur_step,
        )
        while self.cur_step < self.cfg.train.n_steps:
            # get a batch of data
            batch = next(train_data_generator)
            # run a step
            train_status = self.run_step_D(batch)
            self.status_tracker.track_status('Train', train_status, self.cur_step)
            train_status = self.run_step_G(batch)
            self.status_tracker.track_status('Train', train_status, self.cur_step)
            # save checkpoint
            if check_freq(self.cfg.train.save_freq, self.cur_step):
                self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step:0>6d}.pt'))
            # sample from current model
            if check_freq(self.cfg.train.sample_freq, self.cur_step):
                self.sample(os.path.join(self.exp_dir, 'samples', f'step{self.cur_step:0>6d}.png'))
            # evaluate
            if check_freq(self.cfg.train.evaluate_freq, self.cur_step):
                eval_status = self.evaluate(self.valid_loader)
                self.status_tracker.track_status('Eval', eval_status, self.cur_step)
            # synchronizes all processes
            if is_dist_avail_and_initialized():
                dist.barrier()
            self.cur_step += 1
        # save the last checkpoint if not saved
        if not check_freq(self.cfg.train.save_freq, self.cur_step - 1):
            self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step-1:0>6d}.pt'))
        self.status_tracker.close()
        self.logger.info('End of training')

    def run_step_G(self, batch):
        self.inpaintnet.train()
        self.refinenet.train()
        self.pdisc.train()
        self.optimizer_inpaintnet.zero_grad()
        self.optimizer_refinenet.zero_grad()
        self.optimizer_classifier.zero_grad()
        self.loss_meter_gen.reset()
        batch_X, batch_gt_img, batch_noise, batch_mask = batch
        batch_size = batch_X.shape[0]
        for i in range(0, batch_size, self.micro_batch):
            X = batch_X[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            gt_img = batch_gt_img[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            mask = batch_mask[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            # no need to synchronize gradient before the last micro batch
            no_sync = is_dist_avail_and_initialized() and (i + self.micro_batch) < batch_size
            cm1 = self.inpaintnet.no_sync() if no_sync else nullcontext()
            cm2 = self.refinenet.no_sync() if no_sync else nullcontext()
            cm3 = self.classifier.no_sync() if no_sync else nullcontext()
            with cm1, cm2, cm3:
                recX, projs, conf_mask_hier = self.inpaintnet(X, mask)
                if self.args.phase == 1:
                    refX = self.refinenet(recX.detach(), mask)
                else:
                    pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
                    refX = self.refinenet(recX.detach(), pred_mask)

                # hierarchical contrastive loss
                mask_resized = F.interpolate(mask, size=conf_mask_hier['pred_masks'][0].shape[-2:])
                losses_contrast = self.hier_contrast(
                    features=projs,
                    valid_masks=conf_mask_hier['valid_masks'],
                    invalid_masks=conf_mask_hier['invalid_masks'],
                    confs=conf_mask_hier['confs'],
                    acc_valid_masks=conf_mask_hier['acc_valid_masks'],
                    acc_invalid_masks=conf_mask_hier['acc_invalid_masks'],
                    gt_mask=mask_resized,
                )
                loss_contrast = sum(losses_contrast)
                # reconstruction loss
                loss_rec = self.l1(recX, gt_img)
                loss_rec_refined = self.l1(refX, gt_img)
                # perceptual loss
                loss_perc = self.perceptual(recX, gt_img)
                loss_perc_refined = self.perceptual_refined(refX, gt_img)
                # adversarial loss
                loss_adv_refined = self.patch_adv.forward_G(fakeX=refX)
                # classifier loss
                scores = self.classifier(conf_mask_hier['centers'][-1]).squeeze(-1)
                loss_cls = (self.bcewithlogits(scores[:, 0], torch.ones((recX.shape[0], ), device=self.device)) +
                            self.bcewithlogits(scores[:, 1], torch.zeros((recX.shape[0], ), device=self.device))) / 2
                # total
                loss_total = (loss_contrast * self.cfg.train.lambda_contrast +
                              loss_rec * self.cfg.train.lambda_rec +
                              loss_rec_refined * self.cfg.train.lambda_rec_refined +
                              loss_perc * self.cfg.train.lambda_perc +
                              loss_perc_refined * self.cfg.train.lambda_perc_refined +
                              loss_adv_refined * self.cfg.train.lambda_adv_refined +
                              loss_cls)
                loss_total = loss_total * X.shape[0] / batch_size
                loss_total.backward()
            self.loss_meter_gen.update(kvs=dict(
                loss_rec=loss_rec.detach(),
                loss_rec_refined=loss_rec_refined.detach(),
                loss_perc=loss_perc.detach(),
                loss_perc_refined=loss_perc_refined.detach(),
                loss_adv_refined=loss_adv_refined.detach(),
                loss_contrast=loss_contrast.detach(),
                loss_cls=loss_cls.detach(),
            ), n=X.shape[0])
        train_status = self.loss_meter_gen.compute()
        self.optimizer_inpaintnet.step()
        self.optimizer_refinenet.step()
        self.optimizer_classifier.step()
        return train_status

    def run_step_D(self, batch):
        self.pdisc.train()
        self.optimizer_pdisc.zero_grad()
        self.loss_meter_pdisc.reset()
        batch_X, batch_gt_img, batch_noise, batch_mask = batch
        batch_size = batch_X.shape[0]
        for i in range(0, batch_size, self.micro_batch):
            X = batch_X[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            gt_img = batch_gt_img[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            mask = batch_mask[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            # no need to synchronize gradient before the last micro batch
            no_sync = is_dist_avail_and_initialized() and (i + self.micro_batch) < batch_size
            cm = self.pdisc.no_sync() if no_sync else nullcontext()
            with cm:
                recX, projs, conf_mask_hier = self.inpaintnet(X, mask)
                if self.args.phase == 1:
                    refX = self.refinenet(recX.detach(), mask)
                else:
                    pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
                    refX = self.refinenet(recX.detach(), pred_mask)
                lossD = self.patch_adv.forward_D(realX=gt_img, fakeX=refX.detach())
                lossD.backward()
            self.loss_meter_pdisc.update(kvs=dict(lossD=lossD.detach()), n=X.shape[0])
        train_status = self.loss_meter_pdisc.compute()
        self.optimizer_pdisc.step()
        return train_status

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.inpaintnet.eval()
        self.refinenet.eval()
        self.metric_image.reset()
        self.metric_mask.reset()
        pbar = tqdm.tqdm(dataloader, desc='Evaluating', leave=False, disable=not is_main_process())
        for X, gt_img, noise, mask in pbar:
            X = X.to(device=self.device, dtype=torch.float32)
            gt_img = gt_img.to(device=self.device, dtype=torch.float32)
            mask = mask.to(device=self.device, dtype=torch.float32)
            recX, projs, conf_mask_hier = self.inpaintnet(X, mask)
            if self.args.phase == 1:
                refX = self.refinenet(recX, mask)
            else:
                pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
                refX = self.refinenet(recX, pred_mask)
            self.metric_mask.update(
                F.interpolate(conf_mask_hier['acc_confs'][0], size=mask.shape[-2:]), mask.long(),
            )
            self.metric_image.update((refX + 1) / 2, (gt_img + 1) / 2)
        pbar.close()
        eval_status = self.metric_image.compute()
        eval_status.update({
            k: v.mean()
            for k, v in self.metric_mask.compute().items()
        })
        return eval_status

    @main_process_only
    @torch.no_grad()
    def sample(self, savepath: str):
        inpaintnet = get_bare_model(self.inpaintnet)
        refinenet = get_bare_model(self.refinenet)
        inpaintnet.eval()
        refinenet.eval()
        show_imgs = []
        for i in tqdm.tqdm(range(6), desc='Sampling', leave=False, disable=not is_main_process()):
            X, gt_img, noise, mask = self.valid_set[i]
            X = X.to(device=self.device, dtype=torch.float32).unsqueeze(0)
            gt_img = gt_img.to(device=self.device, dtype=torch.float32).unsqueeze(0)
            mask = mask.to(device=self.device, dtype=torch.float32).unsqueeze(0)

            recX, projs, conf_mask_hier = inpaintnet(X, mask)
            if self.args.phase == 1:
                refX = refinenet(recX, mask)
            else:
                pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
                refX = refinenet(recX, pred_mask)

            shape = (self.cfg.data.img_size, self.cfg.data.img_size)
            pred_masks = [F.interpolate(m.float().cpu(), size=shape) for m in conf_mask_hier['pred_masks']]

            show_imgs.append(((gt_img[0] + 1) / 2).cpu())
            show_imgs.append(((X[0] + 1) / 2).cpu())
            show_imgs.append(mask[0].repeat(3, 1, 1).cpu())
            for j in range(len(projs)):
                show_imgs.append(pred_masks[j][0].repeat(3, 1, 1).cpu())
            show_imgs.append(((refX[0] + 1) / 2).cpu())
        show_imgs = torch.stack(show_imgs, dim=0)
        save_image(show_imgs, savepath, nrow=len(show_imgs) // 6, normalize=True, value_range=(0, 1))
