import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from functools import partial
from inspect import isfunction
import numpy as np
import json 
import os

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        in_dims = hparams['hidden_size']
        self.encoder_hidden = hparams['hidden_size']
        self.residual_layers = hparams['residual_layers']
        self.residual_channels = hparams['residual_channels']
        self.dilation_cycle_length = hparams['dilation_cycle_length']

        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.encoder_hidden, self.residual_channels, 2 ** (i % self.dilation_cycle_length))
            for i in range(self.residual_layers)
        ])
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        return x[:, None, :, :]

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - np.exp(2. * log_mean_coeff)
    return var

def sigma_beta_schedule(timesteps, min_beta=0.01, max_beta=20, use_geometric=False):
    eps_small = 1e-3
   
    t = np.arange(0, timesteps + 1, dtype=np.float64)
    t = t / timesteps
    t = t * (1. - eps_small) + eps_small
    
    if use_geometric:
        var = var_func_geometric(t, min_beta, max_beta)
    else:
        var = var_func_vp(t, min_beta, max_beta)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = np.array([1e-8])
    betas = np.concatenate((first, betas))
    sigmas = betas**0.5
    a_s = np.sqrt(1-betas)
    return sigmas, a_s, betas

def linear_beta_schedule(timesteps, max_beta=0.01):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp),
}

class DiffusionProsodyGenerator(nn.Module):
    def __init__(self, hparams, out_dims=None):
        super().__init__()
        self.hparams = hparams
        out_dims = hparams['hidden_size']
        denoise_fn = DIFF_DECODERS[hparams['diff_decoder_type']](hparams)
        timesteps = hparams['timesteps']
        K_step = hparams['K_step']
        loss_type = hparams['diff_loss_type']

        stats_f = os.path.join(hparams['tts_model'], 
                "stats_lpv_{}.json".format(hparams['train_set_name']))
                    
        with open(stats_f) as f:
            stats = json.load(f)
        spec_min = stats['lpv_min'][0]
        spec_max = stats['lpv_max'][0]

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims

        sigmas, a_s, betas = sigma_beta_schedule(timesteps, hparams['min_beta'], hparams['max_beta'])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
        self.loss_type = loss_type
        self.proj = nn.Linear(384, 192)

        a_s_cum = np.cumprod(a_s)
        sigmas_cum = np.sqrt(1 - a_s_cum ** 2)
        a_s_prev = np.copy(a_s)
        a_s_prev[-1] = 1

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('a_s', to_torch(a_s))
        self.register_buffer('sigmas', to_torch(sigmas))

        self.register_buffer('a_s_prev', to_torch(a_s_prev))
        self.register_buffer('a_s_cum', to_torch(a_s_cum))
        self.register_buffer('sigmas_cum', to_torch(sigmas_cum))

        # Posterior coefficients
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas, 0)
        alphas_cumprod_prev = np.concatenate((np.array([1.]), alphas_cumprod[:-1]))
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('posterior_mean_coef1', to_torch((betas * np.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))))
        self.register_buffer('posterior_mean_coef2', to_torch((1 - alphas_cumprod_prev) * np.sqrt(alphas) / (1 - alphas_cumprod)))
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', torch.log(to_torch(posterior_variance).clamp(min=1e-20)))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False):
        b, *_, device = *x_start.shape, x_start.device
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = noise_like(x_start.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.a_s_cum, t, x_start.shape) * x_start +
                extract(self.sigmas_cum, t, x_start.shape) * noise
        )

    def q_sample_pairs(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_plus_one = extract(self.a_s, t+1, x_start.shape) * x_start + \
                extract(self.sigmas, t+1, x_start.shape) * noise

        return x_t, x_t_plus_one

    def forward(self, word_tokens, spk_embed=None, spk_id=None, lpv=None,
                ph2word=None, infer=False, padding=None, noise_scale=1.0, **kwargs):
        b, *_, device = *word_tokens.shape, word_tokens.device

        cond = (word_tokens + spk_embed.unsqueeze(1)).transpose(1, 2)

        ret = {}
        if padding is not None:
            padding = padding.unsqueeze(1).unsqueeze(1)

        if not infer:
            t = torch.randint(0, self.K_step, (b,), device=device).long()
            x = lpv
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            
            noise = default(None, lambda: torch.randn_like(x))

            x_t, x_tp1 = self.q_sample_pairs(x_start=x, t=t, noise=noise)
            # print(x_t.shape)
            x_0_predict = self.denoise_fn(x_tp1, t, cond)
            x_0_predict.clamp_(-1., 1.)
            x_pos_sample = self.q_posterior_sample(x_0_predict, x_tp1, t)
            
            x_0_predict = x_0_predict * padding
            x_t = x_t * padding
            x_tp1 = x_tp1 * padding
            x_pos_sample = x_pos_sample * padding

            x_0_predict = x_0_predict[:, 0].transpose(1, 2)
            x_t = x_t[:, 0].transpose(1, 2)
            x_tp1 = x_tp1[:, 0].transpose(1, 2)
            x_pos_sample = x_pos_sample[:, 0].transpose(1, 2)
            
            ret["x_0_predict"] = x_0_predict
            ret["x_t"] = x_t
            ret["x_tp1"] = x_tp1
            ret["x_pos_sample"] = x_pos_sample
            ret["t"] = t
            ret["cond"] = cond
            ret['lpv_out'] = None

        else:
            t = self.K_step
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = torch.randn(shape, device=device) * noise_scale

            for i in reversed(range(0, t)):
                t_i = torch.full((x.size(0),), i, device=device).long()
                x_0 = self.denoise_fn(x, t_i, cond=cond)
                x_new = self.q_posterior_sample(x_0, x, t_i)
                x = x_new.detach()

            x = x[:, 0].transpose(1, 2)
            ret['lpv_out'] = self.denorm_spec(x)
        return ret

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min