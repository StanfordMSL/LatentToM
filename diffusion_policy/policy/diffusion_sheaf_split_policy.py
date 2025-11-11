from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.sheaf_obs_encoder import SheafObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class ConfidenceModule(torch.nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.shared_fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(torch.nn.Linear(256, 1), torch.nn.Sigmoid())

    def forward(self, x):
        feat_shared = self.shared_fc(x)
        conf = self.head(feat_shared)
        return conf

class ToMCrossPredictor(nn.Module):
    def __init__(self, shared_dim=1024, private_dim=1038):
        super().__init__()
        self.query_fc = nn.Linear(shared_dim, 512)
        self.key_fc = nn.Linear(private_dim, 512)
        self.value_fc = nn.Linear(private_dim, 512)

        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.out_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, private_dim)
        )

    def forward(self, shared_embedding, candidate_private):
        # shared_embedding: [B, shared_dim]
        # candidate_private: [B, private_dim]

        query = self.query_fc(shared_embedding).unsqueeze(1)          # [B,1,512]
        key = self.key_fc(candidate_private).unsqueeze(1)             # [B,1,512]
        value = self.value_fc(candidate_private).unsqueeze(1)         # [B,1,512]

        attn_output, _ = self.attn(query, key, value)                 # [B,1,512]
        attn_output = attn_output.squeeze(1)                          # [B,512]

        pred_private = self.out_fc(attn_output)                       # [B,private_dim]
        return pred_private


class DiffusionSheafSplitPolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: SheafObsEncoder,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 # parameters passed to step
                 **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['arm1_action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0][0]  # 1031 for decentralized
        sheaf_embedding_dim = obs_encoder.output_shape()[1][0]
        # new obs combined feature dim
        obs_feature_dim = obs_feature_dim + sheaf_embedding_dim

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # for loss
        self.tom_predictor = ToMCrossPredictor()

        # confidence module
        self.confidence_module = ConfidenceModule()

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t,
                                 local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       return_embedding_only=False,
                       encoded_obs=None) -> Dict[str, torch.Tensor]:
        device = self.device
        dtype = self.dtype
        Da = self.action_dim
        To = self.n_obs_steps
        T = self.horizon
        Do = self.obs_feature_dim

        local_cond = None
        global_cond = None

        if encoded_obs is None:
            arm_action_idx = 'null'
            if 'camera_1' in obs_dict.keys():
                arm_action_idx = 'arm1_action'
            if 'camera_4' in obs_dict.keys():
                arm_action_idx = 'arm2_action'
            assert 'past_action' not in obs_dict  # not implemented yet
            # normalize input
            nobs = self.normalizer.normalize(obs_dict)
            value = next(iter(nobs.values()))
            B, To = value.shape[:2]

            # handle different ways of passing observation
            if self.obs_as_global_cond:
                # condition through global feature
                this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                nobs_features, sheaf_embedding = self.obs_encoder(this_nobs)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
                # reshape back sheaf_embedding
                sheaf_embedding = sheaf_embedding.reshape(B, -1)
                # empty data for action
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                # condition through impainting
                this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                nobs_features, sheaf_embedding = self.obs_encoder(this_nobs)
                # reshape back to B, T, Do
                nobs_features = nobs_features.reshape(B, To, -1)
                # reshape back sheaf_embedding
                sheaf_embedding = sheaf_embedding.reshape(B, To, -1)
                cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:, :To, Da:] = nobs_features
                cond_mask[:, :To, Da:] = True

            encoded_obs={
                'global_cond': global_cond,
                'sheaf_embedding': sheaf_embedding,
                'cond_data': cond_data,
                'cond_mask': cond_mask,
                'arm_action_idx': arm_action_idx
            }
        else:
            global_cond = encoded_obs['global_cond']
            cond_data = encoded_obs['cond_data']
            cond_mask = encoded_obs['cond_mask']
            arm_action_idx = encoded_obs['arm_action_idx']

        # at first stage, we do not need to predict the action
        if return_embedding_only:
            return encoded_obs

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer[arm_action_idx].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        arm_action_idx = 'null'
        if 'camera_1' in batch['obs']:
            arm_action_idx = 'arm1_action'
        if 'camera_4' in batch['obs']:
            arm_action_idx = 'arm2_action'
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(
            batch['obs'])  # after normalization, the size of obs is same [B, 2, 3, 240, 320] and [B, 2, 3/4]
        nactions = self.normalizer[arm_action_idx].normalize(
            batch['action'])  # [B, horizon, 10], 3 pos + 6d orientation + 1 gripper
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # # check the size of the normalized obs and actions
        # for key, value in nobs.items():
        #     try:
        #         print(f"Key: {key}, Shape: {value.shape}")
        #     except AttributeError:
        #         print(f"Key: {key}, Type: {type(value)} (No shape attribute)")
        # print(f"Key: action, Shape: {nactions.shape}")
        # print(f"Batch size: {batch_size}, Horizon: {horizon}")

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:  # in our case, self.obs_as_global_cond is True
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            # # After the above code, the size of this_nobs is [B*2, 3, 240, 320] and [B*2, 3/4]
            # for key, value in this_nobs.items():
            #     try:
            #         print(f"this_nobs Key: {key}, Shape: {value.shape}")
            #     except AttributeError:
            #         print(f"this_nobs Key: {key}, Type: {type(value)} (No shape attribute)")
            # nobs_features: this is private embedding, sheaf_embedding: this is shared embedding
            nobs_features, sheaf_embedding = self.obs_encoder(this_nobs)  # node features: [batch_size * 2, 1031]
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)  # [batch_size, 2062]
            private_embedding = global_cond.clone()
            # reshape back sheaf_embedding: sheaf embedding is private embedding
            sheaf_embedding = sheaf_embedding.reshape(batch_size, -1)
            # combine global cond and sheaf embedding in the final dim
            global_cond = torch.cat([global_cond, sheaf_embedding], dim=-1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features, sheaf_embedding = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            private_embedding = nobs_features.clone()
            # reshape back sheaf_embedding
            sheaf_embedding = sheaf_embedding.reshape(batch_size, horizon, -1)  # [batch_size, 2, 1031]
            # combine global cond and sheaf embedding in the final dim
            nobs_features = torch.cat([nobs_features, sheaf_embedding], dim=-1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        # the condition mask and the trajectory have the same size [B, 100, 20]

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss, sheaf_embedding, private_embedding



