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
from diffusion_policy.policy.diffusion_individual_camera_policy import DiffusionIndividualCameraPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to, build_arm_sub_batch_individual_cam
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import torch.nn.functional as F

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionIndividualCameraWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.arm1_model: DiffusionIndividualCameraPolicy = hydra.utils.instantiate(cfg.policy)
        self.arm2_model: DiffusionIndividualCameraPolicy = hydra.utils.instantiate(cfg.policy)

        self.arm1_ema_model: DiffusionIndividualCameraPolicy = None
        self.arm2_ema_model: DiffusionIndividualCameraPolicy = None
        if cfg.training.use_ema:
            self.arm1_ema_model = copy.deepcopy(self.arm1_model)
            self.arm2_ema_model = copy.deepcopy(self.arm2_model)

        self.optimizer_arm1 = hydra.utils.instantiate(
            cfg.optimizer, params=self.arm1_model.parameters())
        self.optimizer_arm2 = hydra.utils.instantiate(
            cfg.optimizer, params=self.arm2_model.parameters())

        self.global_step = 0
        self.epoch = 0

        self.wandb_launch = False  # set as True when do the actual training

    def confidence_entropy_loss(self, confidence):
        epsilon = 1e-8
        entropy = - (confidence * torch.log(confidence + epsilon) + (1 - confidence) * torch.log(
            1 - confidence + epsilon))
        return entropy.mean()

    def compute_confidence_one_way_loss(self, embedding_arm1, embedding_arm2, confidence_arm1, confidence_arm2):
        lambda_entropy = 0.01
        conf_diff = torch.abs(confidence_arm1 - confidence_arm2).detach()
        mask = (confidence_arm1 >= confidence_arm2).float()

        loss1 = F.mse_loss(embedding_arm2, embedding_arm1.detach(), reduction='none')
        loss2 = F.mse_loss(embedding_arm1, embedding_arm2.detach(), reduction='none')

        loss = (mask * loss1 + (1 - mask) * loss2) * (1 + conf_diff)
        loss = loss.mean()

        entropy_loss = self.confidence_entropy_loss(confidence_arm1) + self.confidence_entropy_loss(confidence_arm2)
        total_loss = loss + lambda_entropy * entropy_loss

        return total_loss

    def compute_sheaf_loss(self, shared_arm1, shared_arm2, private_arm1, private_arm2, model_arm1=None, model_arm2=None):
        """
        adjust these three parameters as you wish
        """
        alpha = 1.0
        beta = 0.5
        gamma = 0.5

        mse_loss = torch.nn.functional.mse_loss(shared_arm1, shared_arm2)
        conf_arm1 = model_arm1.confidence_module(shared_arm1)
        conf_arm2 = model_arm2.confidence_module(shared_arm2)
        conf_loss = self.compute_confidence_one_way_loss(shared_arm1, shared_arm2, conf_arm1, conf_arm2)
        pred_private2_by_arm1 = model_arm1.tom_predictor(shared_arm1, private_arm2)
        pred_private1_by_arm2 = model_arm2.tom_predictor(shared_arm2, private_arm1)
        cross_pred_loss = F.mse_loss(pred_private2_by_arm1, private_arm2.detach()) + \
                          F.mse_loss(pred_private1_by_arm2, private_arm1.detach())

        loss = alpha * mse_loss + beta * conf_loss + gamma * cross_pred_loss
        return loss

    def sheaf_laplacian_online_adjustment(self, embedding_arm1, embedding_arm2, gamma=0.1, steps=3):
        for _ in range(steps):
            delta = embedding_arm1 - embedding_arm2
            embedding_arm1 = embedding_arm1 - gamma * delta
            embedding_arm2 = embedding_arm2 + gamma * delta
        return embedding_arm1, embedding_arm2

    def process_encoded_obs(self, encoded_obs, sheaf_embedding):
        processed_encoded_obs = encoded_obs.copy()
        processed_global_cond = torch.cat([encoded_obs['global_cond'], sheaf_embedding], dim=-1)
        processed_encoded_obs['global_cond'] = processed_global_cond
        return processed_encoded_obs

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            arm1_ckpt_path = pathlib.Path(self.output_dir).joinpath('checkpoints', 'arm1_latest.ckpt')
            arm2_ckpt_path = pathlib.Path(self.output_dir).joinpath('checkpoints', 'arm2_latest.ckpt')
            if arm1_ckpt_path.is_file():
                print(f"Resuming arm1 from checkpoint {arm1_ckpt_path}")
                self.load_checkpoint(path=arm1_ckpt_path, arm="arm1")
            if arm2_ckpt_path.is_file():
                print(f"Resuming arm2 from checkpoint {arm2_ckpt_path}")
                self.load_checkpoint(path=arm2_ckpt_path, arm="arm2")

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.arm1_model.set_normalizer(normalizer)
        self.arm2_model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.arm1_ema_model.set_normalizer(normalizer)
            self.arm2_ema_model.set_normalizer(normalizer)

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
            last_epoch=self.global_step-1
        )

        ema_arm1: EMAModel = None
        ema_arm2: EMAModel = None
        if cfg.training.use_ema:
            ema_arm1 = hydra.utils.instantiate(
                cfg.ema,
                model=self.arm1_ema_model)
            ema_arm2 = hydra.utils.instantiate(
                cfg.ema,
                model=self.arm2_ema_model)

        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

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

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.arm1_model.to(device)
        self.arm2_model.to(device)
        if self.arm1_ema_model is not None:
            self.arm1_ema_model.to(device)
        if self.arm2_ema_model is not None:
            self.arm2_ema_model.to(device)
        optimizer_to(self.optimizer_arm1, device)
        optimizer_to(self.optimizer_arm2, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
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
                        ---------------------------------------------
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
                        ---------------------------------------------
                        For individual camera task:
                        batch = {
                        obs: {
                            camera1: B, To, C, H, W
                            camera2: B, To, C, H, W
                            camera3: B, To, C, H, W
                            camera4: B, To, C, H, W
                            arm1_robot_eef_pos: B, To, 3
                            arm1_eef_quat: B, To, 4
                            arm2_robot_eef_pos: B, To, 3
                            arm2_eef_quat: B, To, 4
                            }
                        arm1_action: {B, T, 10}
                        arm2_action: {B, T, 10}
                        }
                        """
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        arm1_batch = build_arm_sub_batch_individual_cam(batch, arm_id=1)
                        arm2_batch = build_arm_sub_batch_individual_cam(batch, arm_id=2)

                        raw_loss_arm1, sheaf_embedding_arm1, private_arm1 = self.arm1_model.compute_loss(arm1_batch)
                        loss_arm1 = raw_loss_arm1 / cfg.training.gradient_accumulate_every
                        raw_loss_arm2, sheaf_embedding_arm2, private_arm2 = self.arm2_model.compute_loss(arm2_batch)
                        loss_arm2 = raw_loss_arm2 / cfg.training.gradient_accumulate_every

                        raw_loss_sheaf = self.compute_sheaf_loss(sheaf_embedding_arm1, sheaf_embedding_arm2, private_arm1, private_arm2, self.arm1_model, self.arm2_model)
                        loss_sheaf = raw_loss_sheaf / cfg.training.gradient_accumulate_every
                        weighted_loss_sheaf = 0.5 * loss_sheaf

                        loss_arm1_total = loss_arm1 + weighted_loss_sheaf
                        loss_arm1_total.backward(retain_graph=True)
                        loss_arm2_total = loss_arm2 + weighted_loss_sheaf
                        loss_arm2_total.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            for opt, scheduler in [
                                (self.optimizer_arm1, lr_scheduler_arm1),
                                (self.optimizer_arm2, lr_scheduler_arm2)
                            ]:
                                opt.step()
                                opt.zero_grad()
                                scheduler.step()

                        if cfg.training.use_ema:
                            ema_arm1.step(self.arm1_model)
                            ema_arm2.step(self.arm2_model)

                        raw_loss_arm1_cpu = raw_loss_arm1.item()
                        tepoch.set_postfix(loss=raw_loss_arm1_cpu, refresh=False)
                        train_losses_arm1.append(raw_loss_arm1_cpu)
                        raw_loss_arm2_cpu = raw_loss_arm2.item()
                        tepoch.set_postfix(loss=raw_loss_arm2_cpu, refresh=False)
                        train_losses_arm2.append(raw_loss_arm2_cpu)
                        raw_loss_sheaf_cpu = raw_loss_sheaf.item()
                        tepoch.set_postfix(loss=raw_loss_sheaf_cpu, refresh=False)
                        train_losses_sheaf.append(raw_loss_sheaf_cpu)
                        step_log = {
                            'train_loss_arm1': raw_loss_arm1_cpu,
                            'train_loss_arm2': raw_loss_arm2_cpu,
                            'train_loss_sheaf': raw_loss_sheaf_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr_arm1': lr_scheduler_arm1.get_last_lr()[0],
                            'lr_arm2': lr_scheduler_arm2.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            if self.wandb_launch:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                train_loss_arm1 = np.mean(train_losses_arm1)
                train_loss_arm2 = np.mean(train_losses_arm2)
                train_loss_sheaf = np.mean(train_losses_sheaf)
                step_log['train_loss_arm1'] = train_loss_arm1
                step_log['train_loss_arm2'] = train_loss_arm2
                step_log['train_loss_sheaf'] = train_loss_sheaf

                arm1_policy = self.arm1_model
                arm2_policy = self.arm2_model
                if cfg.training.use_ema:
                    arm1_policy = self.arm1_ema_model
                    arm2_policy = self.arm2_ema_model
                arm1_policy.eval()
                arm2_policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log_arm1 = env_runner.run(arm1_policy)
                    runner_log_arm2 = env_runner.run(arm2_policy)
                    step_log.update(runner_log_arm1)
                    step_log.update(runner_log_arm2)

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses_arm1 = list()
                        val_losses_arm2 = list()
                        val_losses_sheaf = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                arm1_batch = build_arm_sub_batch_individual_cam(batch, arm_id=1)
                                arm2_batch = build_arm_sub_batch_individual_cam(batch, arm_id=2)

                                loss_arm1, val_sheaf_embedding_arm1 = self.arm1_model.compute_loss(arm1_batch)
                                val_losses_arm1.append(loss_arm1.item())
                                loss_arm2, val_sheaf_embedding_arm2 = self.arm2_model.compute_loss(arm2_batch)
                                val_losses_arm2.append(loss_arm2.item())

                                loss_sheaf = self.compute_sheaf_loss(val_sheaf_embedding_arm1, val_sheaf_embedding_arm2)
                                weighted_loss_sheaf = loss_sheaf + cfg.training.sheaf_loss_weight
                                val_losses_sheaf.append(weighted_loss_sheaf.item())

                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
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

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))

                        arm1_batch = build_arm_sub_batch_individual_cam(batch, arm_id=1)
                        arm2_batch = build_arm_sub_batch_individual_cam(batch, arm_id=2)

                        arm1_obs_dict = arm1_batch['obs']
                        arm2_obs_dict = arm2_batch['obs']

                        arm1_gt_action = arm1_batch['action']
                        arm2_gt_action = arm2_batch['action']


                        arm1_encoded_obs = arm1_policy.predict_action(arm1_obs_dict, return_embedding_only=True, encoded_obs=None)
                        arm2_encoded_obs = arm2_policy.predict_action(arm2_obs_dict, return_embedding_only=True, encoded_obs=None)
                        arm1_sheaf_embedding = arm1_encoded_obs['sheaf_embedding']
                        arm2_sheaf_embedding = arm2_encoded_obs['sheaf_embedding']

                        """
                        Here we can do online adjustment of the sheaf embeddings before feeding into the diffusion model
                        """
                        # arm1_sheaf_embedding_processed, arm2_sheaf_embedding_processed = self.sheaf_laplacian_online_adjustment(arm1_sheaf_embedding, arm2_sheaf_embedding)
                        arm1_sheaf_embedding_processed = arm1_sheaf_embedding
                        arm2_sheaf_embedding_processed = arm2_sheaf_embedding

                        arm1_encoded_obs_processed = self.process_encoded_obs(arm1_encoded_obs, arm1_sheaf_embedding_processed)
                        arm2_encoded_obs_processed = self.process_encoded_obs(arm2_encoded_obs, arm2_sheaf_embedding_processed)

                        result_arm1 = arm1_policy.predict_action(obs_dict=None, return_embedding_only=False, encoded_obs=arm1_encoded_obs_processed)
                        result_arm2 = arm2_policy.predict_action(obs_dict=None, return_embedding_only=False, encoded_obs=arm2_encoded_obs_processed)

                        pred_action_arm1 = result_arm1["action_pred"]
                        pred_action_arm2 = result_arm2["action_pred"]

                        mse_arm1 = torch.nn.functional.mse_loss(pred_action_arm1, arm1_gt_action)
                        mse_arm2 = torch.nn.functional.mse_loss(pred_action_arm2, arm2_gt_action)

                        step_log["train_action_mse_error_arm1"] = mse_arm1.item()
                        step_log["train_action_mse_error_arm2"] = mse_arm2.item()

                        pred_action_dual = torch.cat([pred_action_arm1, pred_action_arm2], dim=-1)  # [32, 40, 20]
                        gt_action_dual = torch.cat([arm1_gt_action, arm2_gt_action], dim=-1)
                        total_mse = torch.nn.functional.mse_loss(pred_action_dual, gt_action_dual)

                        step_log["train_action_mse_error_total"] = total_mse.item()

                        del batch
                        del arm1_batch
                        del arm2_batch
                        del arm1_obs_dict
                        del arm2_obs_dict
                        del arm1_gt_action
                        del arm2_gt_action
                        del result_arm1
                        del result_arm2
                        del pred_action_arm1
                        del pred_action_arm2
                        del mse_arm1
                        del mse_arm2
                        del total_mse

                if (self.epoch % cfg.training.checkpoint_every) == 0:

                    if cfg.checkpoint.save_last_ckpt:
                        arm1_exclude = tuple(
                            list(self.exclude_keys) + ['arm2_model', 'arm2_ema_model', 'optimizer_arm2'])
                        arm1_ckpt_path = pathlib.Path(self.output_dir).joinpath('checkpoints', 'arm1_latest.ckpt')
                        self.save_checkpoint(path=arm1_ckpt_path, tag='arm1_latest', exclude_keys=arm1_exclude)

                        arm2_exclude = tuple(
                            list(self.exclude_keys) + ['arm1_model', 'arm1_ema_model', 'optimizer_arm1'])
                        arm2_ckpt_path = pathlib.Path(self.output_dir).joinpath('checkpoints', 'arm2_latest.ckpt')
                        self.save_checkpoint(path=arm2_ckpt_path, tag='arm2_latest', exclude_keys=arm2_exclude)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path_arm1, topk_ckpt_path_arm2 = topk_manager.get_ckpt_paths(metric_dict)
                    if topk_ckpt_path_arm1 is not None:
                        arm1_exclude = tuple(list(self.exclude_keys) + ['arm2_model', 'arm2_ema_model', 'optimizer_arm2'])
                        self.save_checkpoint(path=topk_ckpt_path_arm1, tag='arm1_latest', exclude_keys=arm1_exclude)

                    if topk_ckpt_path_arm2 is not None:
                        arm2_exclude = tuple(list(self.exclude_keys) + ['arm1_model', 'arm1_ema_model', 'optimizer_arm1'])
                        self.save_checkpoint(path=topk_ckpt_path_arm2, tag='arm2_latest', exclude_keys=arm2_exclude)

                arm1_policy.train()
                arm2_policy.train()

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
    workspace = TrainDiffusionIndividualCameraWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
