if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_sheaf_image_policy import DiffusionSheafImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionSheafImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model: from 1 arm to 2 arms
        self.arm1_model: DiffusionSheafImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.arm2_model: DiffusionSheafImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.arm1_ema_model: DiffusionSheafImagePolicy = None
        self.arm2_ema_model: DiffusionSheafImagePolicy = None

        if cfg.training.use_ema:
            self.arm1_ema_model = copy.deepcopy(self.arm1_model)
            self.arm2_ema_model = copy.deepcopy(self.arm2_model)

        # define the separate optimizers for each arm
        self.optimizer_arm1 = hydra.utils.instantiate(
            cfg.optimizer, params=self.arm1_model.parameters())
        self.optimizer_arm2 = hydra.utils.instantiate(
            cfg.optimizer, params=self.arm2_model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # are we debugging or not?
        self.wandb_launch = True

    def compute_sheaf_loss(self, embedding_arm1, embedding_arm2):
        # embedding_arm1 and embedding_arm2 are the output of the models
        loss = torch.nn.functional.mse_loss(embedding_arm1, embedding_arm2)
        return loss

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training - todo: this is only for one arm
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.arm1_model.set_normalizer(normalizer)
        self.arm2_model.set_normalizer(normalizer)

        if cfg.training.use_ema:
            self.arm1_ema_model.set_normalizer(normalizer)
            self.arm2_ema_model.set_normalizer(normalizer)

        # configure lr scheduler for two arms' optimizer separately
        lr_scheduler_arm1 = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer_arm1,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )
        lr_scheduler_arm2 = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer_arm2,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema for two arms
        ema_arm1: EMAModel = None
        ema_arm2: EMAModel = None
        if cfg.training.use_ema:
            ema_arm1 = hydra.utils.instantiate(
                cfg.ema,
                model=self.arm1_ema_model)
            ema_arm2 = hydra.utils.instantiate(
                cfg.ema,
                model=self.arm2_ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        if self.wandb_launch:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.arm1_model.to(device)
        self.arm2_model.to(device)
        if self.arm1_ema_model is not None:
            self.arm1_ema_model.to(device)
        if self.arm2_ema_model is not None:
            self.arm2_ema_model.to(device)
        optimizer_to(self.optimizer_arm1, device)
        optimizer_to(self.optimizer_arm2, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        # Create a log file path
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            # start loop for training, we need to train 600 epochs
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.arm1_model.obs_encoder.eval()
                    self.arm1_model.obs_encoder.requires_grad_(False)
                    self.arm2_model.obs_encoder.eval()
                    self.arm2_model.obs_encoder.requires_grad_(False)

                train_losses_arm1 = list()
                train_losses_arm2 = list()
                train_losses_sheaf = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # # print batch's data structure
                        # for key, value in batch.items():
                        #     print(f"Key: {key}")
                        #     if isinstance(value, dict):
                        #         for sub_key, sub_value in value.items():
                        #             print(f"  Sub-key: {sub_key}, Shape: {sub_value.shape}")
                        #     else:
                        #         print(f"  Shape: {value.shape}")
                        """
                        For the multi-arm task:
                        batch = {
                        obs: {
                            camera1: B, To, C, H, W
                            camera3: B, To, C, H, W
                            camera4: B, To, C, H, W
                            arm1_robot_eef_pos: B, To, 3
                            arm1_eef_quat: B, To, 4
                            arm2_robot_eef_pos: B, To, 3
                            arm2_eef_quat: B, To, 4
                            }
                        action: {B, T, 20}
                        }
                        """
                        # device transfer, move all batches to GPU
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # split the batch into two arms
                        batch_arm1 = {
                            "obs": {
                                "camera_1": batch["obs"]["camera_1"],
                                "camera_3": batch["obs"]["camera_3"],
                                "arm1_robot_eef_pos": batch["obs"]["arm1_robot_eef_pos"],
                                "arm1_eef_quat": batch["obs"]["arm1_eef_quat"],
                            },
                            "action": batch["action"][:, :, :10]
                        }

                        batch_arm2 = {
                            "obs": {
                                "camera_4": batch["obs"]["camera_4"],
                                "camera_3": batch["obs"]["camera_3"],
                                "arm2_robot_eef_pos": batch["obs"]["arm2_robot_eef_pos"],
                                "arm2_eef_quat": batch["obs"]["arm2_eef_quat"],
                            },
                            "action": batch["action"][:, :, 10:]
                        }

                        # compute loss for each arm
                        raw_loss_arm1, embedding_arm1 = self.arm1_model.compute_loss_and_embedding(batch_arm1)
                        loss_arm1 = raw_loss_arm1 / cfg.training.gradient_accumulate_every

                        raw_loss_arm2, embedding_arm2 = self.arm2_model.compute_loss_and_embedding(batch_arm2)
                        loss_arm2 = raw_loss_arm2 / cfg.training.gradient_accumulate_every

                        # calculate sheaf loss
                        sheaf_loss = self.compute_sheaf_loss(embedding_arm1, embedding_arm2)
                        sheaf_loss_weighted = cfg.training.sheaf_loss_weight * sheaf_loss

                        # combine the loss and backprop
                        loss_arm1_total = loss_arm1 + sheaf_loss_weighted
                        loss_arm1_total.backward(retain_graph=True)
                        loss_arm2_total = loss_arm2 + sheaf_loss_weighted
                        loss_arm2_total.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer_arm1.step()
                            self.optimizer_arm1.zero_grad()

                            self.optimizer_arm2.step()
                            self.optimizer_arm2.zero_grad()

                            lr_scheduler_arm1.step()
                            lr_scheduler_arm2.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema_arm1.step(self.arm1_model)
                            ema_arm2.step(self.arm2_model)

                        # logging
                        raw_loss_arm1_cpu = raw_loss_arm1.item()
                        tepoch.set_postfix(loss=raw_loss_arm1_cpu, refresh=False)
                        train_losses_arm1.append(raw_loss_arm1_cpu)

                        raw_loss_arm2_cpu = raw_loss_arm2.item()
                        tepoch.set_postfix(loss=raw_loss_arm2_cpu, refresh=False)
                        train_losses_arm2.append(raw_loss_arm2_cpu)

                        sheaf_loss_cpu = sheaf_loss.item()
                        tepoch.set_postfix(sheaf_loss=sheaf_loss_cpu, refresh=False)
                        train_losses_sheaf.append(sheaf_loss_cpu)

                        step_log = {
                            'train_loss_arm1': raw_loss_arm1_cpu,
                            'train_loss_arm2': raw_loss_arm2_cpu,
                            'train_loss_sheaf': sheaf_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr_arm1': lr_scheduler_arm1.get_last_lr()[0],
                            'lr_arm2': lr_scheduler_arm2.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if self.wandb_launch:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss_arm1 = np.mean(train_losses_arm1)
                train_loss_arm2 = np.mean(train_losses_arm2)
                train_loss_sheaf = np.mean(train_losses_sheaf)
                step_log['train_loss_arm1'] = train_loss_arm1
                step_log['train_loss_arm2'] = train_loss_arm2
                step_log['train_loss_sheaf'] = train_loss_sheaf

                # ========= eval for this epoch ==========
                arm1_policy = self.arm1_model
                arm2_policy = self.arm2_model
                if cfg.training.use_ema:
                    arm1_policy = self.arm1_ema_model
                    arm2_policy = self.arm2_ema_model
                arm1_policy.eval()
                arm2_policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log_arm1 = env_runner.run(arm1_policy)
                    runner_log_arm2 = env_runner.run(arm2_policy)
                    # log all
                    step_log.update(runner_log_arm1)
                    step_log.update(runner_log_arm2)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses_arm1 = list()
                        val_losses_arm2 = list()
                        val_losses_sheaf = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                # split the batch into two arms
                                batch_arm1 = {
                                    "obs": {
                                        "camera_1": batch["obs"]["camera_1"],
                                        "camera_3": batch["obs"]["camera_3"],
                                        "arm1_robot_eef_pos": batch["obs"]["arm1_robot_eef_pos"],
                                        "arm1_eef_quat": batch["obs"]["arm1_eef_quat"],
                                    },
                                    "action": batch["action"][:, :, :10]
                                }

                                batch_arm2 = {
                                    "obs": {
                                        "camera_4": batch["obs"]["camera_4"],
                                        "camera_3": batch["obs"]["camera_3"],
                                        "arm2_robot_eef_pos": batch["obs"]["arm2_robot_eef_pos"],
                                        "arm2_eef_quat": batch["obs"]["arm2_eef_quat"],
                                    },
                                    "action": batch["action"][:, :, 10:]
                                }

                                loss_arm1, embedding_arm1 = self.arm1_model.compute_loss_and_embedding(batch_arm1)
                                val_losses_arm1.append(loss_arm1.item())

                                loss_arm2, embedding_arm2 = self.arm2_model.compute_loss_and_embedding(batch_arm2)
                                val_losses_arm2.append(loss_arm2.item())

                                sheaf_loss = self.compute_sheaf_loss(embedding_arm1, embedding_arm2)
                                val_losses_sheaf.append(sheaf_loss.item())

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses_arm1) > 0:
                            val_loss_arm1 = torch.mean(torch.tensor(val_losses_arm1)).item()
                            step_log['val_loss_arm1'] = val_loss_arm1
                        if len(val_losses_arm2) > 0:
                            val_loss_arm2 = torch.mean(torch.tensor(val_losses_arm2)).item()
                            step_log['val_loss_arm2'] = val_loss_arm2
                        if len(val_losses_sheaf) > 0:
                            val_loss_sheaf = torch.mean(torch.tensor(val_losses_sheaf)).item()
                            step_log['val_loss_sheaf'] = val_loss_sheaf

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))

                        obs_arm1 = {
                            "camera_1": batch["obs"]["camera_1"],
                            "camera_3": batch["obs"]["camera_3"],
                            "arm1_robot_eef_pos": batch["obs"]["arm1_robot_eef_pos"],
                            "arm1_eef_quat": batch["obs"]["arm1_eef_quat"],
                        }
                        obs_arm2 = {
                            "camera_4": batch["obs"]["camera_4"],
                            "camera_3": batch["obs"]["camera_3"],
                            "arm2_robot_eef_pos": batch["obs"]["arm2_robot_eef_pos"],
                            "arm2_eef_quat": batch["obs"]["arm2_eef_quat"],
                        }

                        gt_action = batch["action"]
                        gt_action_arm1 = gt_action[:, :, :10]
                        gt_action_arm2 = gt_action[:, :, 10:]

                        result_arm1 = self.arm1_model.predict_action(obs_arm1)
                        result_arm2 = self.arm2_model.predict_action(obs_arm2)

                        pred_action_arm1 = result_arm1["action_pred"]
                        pred_action_arm2 = result_arm2["action_pred"]

                        mse_arm1 = torch.nn.functional.mse_loss(pred_action_arm1, gt_action_arm1)
                        mse_arm2 = torch.nn.functional.mse_loss(pred_action_arm2, gt_action_arm2)

                        step_log["train_action_mse_error_arm1"] = mse_arm1.item()
                        step_log["train_action_mse_error_arm2"] = mse_arm2.item()
                        del batch
                        del obs_arm1
                        del obs_arm2
                        del gt_action
                        del gt_action_arm1
                        del gt_action_arm2
                        del result_arm1
                        del result_arm2
                        del pred_action_arm1
                        del pred_action_arm2
                        del mse_arm1
                        del mse_arm2
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    # # print all keys in metric_dict
                    # for key, value in metric_dict.items():
                    #     print(key)
                    """
                    The current keys are:
                    train_loss_arm1                                                                                                                                                                                                            
                    train_loss_arm2
                    train_loss_sheaf
                    global_step
                    epoch
                    lr_arm1
                    lr_arm2
                    train_action_mse_error_arm1
                    train_action_mse_error_arm2
                    For example:
                    {'train_loss_arm1': 0.15679618404300508, 'train_loss_arm2': 0.17089288150436985, 
                    'train_loss_sheaf': 0.130088009905771, 'global_step': 1822, 'epoch': 0, 
                    'lr_arm1': 9.999998555814473e-05, 'lr_arm2': 9.999998555814473e-05, 
                    'train_action_mse_error_arm1': 6650.9375, 'train_action_mse_error_arm2': 3645.100341796875}
                    """

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # ###########################################################################
                    # # VERSION1 - works
                    # ###########################################################################
                    # if 'train_loss' not in metric_dict:
                    #     if ('train_loss_arm1' in metric_dict and
                    #             'train_loss_arm2' in metric_dict and
                    #             'train_loss_sheaf' in metric_dict):
                    #         metric_dict['train_loss'] = (metric_dict['train_loss_arm1'] +
                    #                                      metric_dict['train_loss_arm2'] +
                    #                                      metric_dict['train_loss_sheaf']) / 3.0
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    #
                    # if topk_ckpt_path is not None:
                    #     # 保存 arm1 的 checkpoint（排除 arm2 相关的变量）
                    #     arm1_exclude = tuple(list(self.exclude_keys) + ['arm2_model', 'arm2_ema_model'])
                    #     arm1_path = pathlib.Path(self.output_dir).joinpath('checkpoints', 'arm1_latest.ckpt')
                    #     self.save_checkpoint(path=arm1_path, tag='arm1_latest', exclude_keys=arm1_exclude)
                    #
                    #     # 保存 arm2 的 checkpoint（排除 arm1 相关的变量）
                    #     arm2_exclude = tuple(list(self.exclude_keys) + ['arm1_model', 'arm1_ema_model'])
                    #     arm2_path = pathlib.Path(self.output_dir).joinpath('checkpoints', 'arm2_latest.ckpt')
                    #     self.save_checkpoint(path=arm2_path, tag='arm2_latest', exclude_keys=arm2_exclude)
                    # ###########################################################################

                    ###########################################################################
                    # Version 2
                    ###########################################################################
                    topk_ckpt_path_arm1, topk_ckpt_path_arm2 = topk_manager.get_ckpt_paths(metric_dict)
                    # print("topk_ckpt_path_arm1", topk_ckpt_path_arm1)
                    # print("topk_ckpt_path_arm2", topk_ckpt_path_arm2)
                    if topk_ckpt_path_arm1 is not None:
                        arm1_exclude = tuple(list(self.exclude_keys) + ['arm2_model', 'arm2_ema_model'])
                        self.save_checkpoint(path=topk_ckpt_path_arm1, tag='arm1_latest', exclude_keys=arm1_exclude)

                    if topk_ckpt_path_arm2 is not None:
                        arm2_exclude = tuple(list(self.exclude_keys) + ['arm1_model', 'arm1_ema_model'])
                        self.save_checkpoint(path=topk_ckpt_path_arm2, tag='arm2_latest', exclude_keys=arm2_exclude)

                # ========= eval end for this epoch ==========
                arm1_policy.train()
                arm2_policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if self.wandb_launch:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionSheafImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
