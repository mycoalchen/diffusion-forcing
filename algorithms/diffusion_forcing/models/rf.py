from typing import Optional, Callable
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .unet3d import Unet3D
from .transformer import Transformer
from .utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, extract, EinopsWrapper

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])

class RectifiedFlow(nn.Module):
    def __init__(
        self,
        x_shape: torch.Size,
        external_cond_dim: int,
        is_causal: bool,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.x_shape = x_shape
        self.external_cond_dim = external_cond_dim
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.clip_noise = cfg.clip_noise
        self.arch = cfg.architecture
        self.stabilization_level = cfg.stabilization_level
        self.is_causal = is_causal
        # NOTE: should guidance_scale be a hyperparameter?
        self.guidance_scale = cfg.guidance_scale

        self._build_model()
        self._build_buffer()

    def _build_model(self):
        x_channel = self.x_shape[0]
        # no video support yet
        assert len(self.x_shape == 1)
        self.model = Transformer(
            x_dim=x_channel,
                external_cond_dim=self.external_cond_dim,
                size=self.arch.network_size,
                num_layers=self.arch.num_layers,
                nhead=self.arch.attn_heads,
                dim_feedforward=self.arch.dim_feedforward,
        )

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    # expects t in [0, 1)
    def z_sample(self, x_start, t, noise=None):  
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        
        return (1 - t) * x_start + t * noise

    # expects noise_levels in [0, self.timesteps)
    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        noise_levels: torch.Tensor,
    ):
        scaled_noise_levels = noise_levels.float() / self.timesteps

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.z_sample(x_start=x, t=scaled_noise_levels, noise=noise)
        v_pred = self.model(noised_x, scaled_noise_levels, external_cond, self.is_causal)
        x_pred = x - scaled_noise_levels * v_pred
        
        v = noise - x

        # TODO: Add loss weighting
        loss = F.mse_loss(v_pred, v.detach(), reduction="none")
        
        return x_pred, loss

    # expects noise_levels in [0, self.sampling_timesteps)
    def sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
    ):
        # NOTE: diffusion.py re-scales from [0, sampling_timesteps] to [-1, timesteps), so 0-noise (certain frame) becomes stabilization_level - 1. Here we instead re-scale from [0, sampling_timesteps] to [0, 1) and replace 0-noise with (stabilization_level - 1) / timesteps.
        scaled_curr_noise_level = curr_noise_level / (self.sampling_timesteps + 1)
        scaled_next_noise_level = next_noise_level / (self.sampling_timesteps + 1)
        clipped_curr_noise_level = torch.where(
            scaled_curr_noise_level == 0,
            torch.full_like(scaled_curr_noise_level, (self.stabilization_level - 1) / self.timesteps),
            scaled_curr_noise_level,
        )

        # NOTE: diffusion.py scales stabilization-level noise by sqrt of alpha_cum so that it has same variance as noisy frames but no added noise. Applying the same reasoning, here we scale by 1 - t
        # NOTE: may cause issues with mixed precision; come back to this if bad things happen
        orig_x = x.clone().detach()
        scaled_context = self.z_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x)
        )
        x = torch.where(self.add_shape_channels(scaled_curr_noise_level == 0), scaled_context, orig_x)

        # NOTE: Following Esser et al. (https://arxiv.org/pdf/2403.03206), use Euler step to get x_pred from v_pred
        v_pred = self.model(x, clipped_curr_noise_level, external_cond, self.is_causal)
        if guidance_fn is not None:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_()    
                score = guidance_fn(x_in)
                grad = torch.autograd.grad(score, x_in)[0]
            v_pred = v_pred + self.guidance_scale * grad
        # this calculation messes up all frames where noise was 0, but that doesn't matter because we mask those out in the next block anyway
        dt = (clipped_curr_noise_level - next_noise_level).view([-1] + [1]*(x.ndim-1))
        x_next = x - dt * v_pred
                
        # only update frames where the noise level decreases
        mask = scaled_curr_noise_level == next_noise_level
        x_next = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_next,
        )

        return x_next