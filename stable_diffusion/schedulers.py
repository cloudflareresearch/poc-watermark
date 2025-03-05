# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from diffusers import DDIMScheduler, DDIMInverseScheduler


def ddim_scheduler():
    """
    Instantiates and returns a DDIM scheduler.
    """
    return DDIMScheduler(
        beta_end=0.012,
        beta_schedule='scaled_linear',
        beta_start=0.00085,
        clip_sample=False,
        set_alpha_to_one =True,
        steps_offset=1,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=False), 'dddim'


def ddim_inverse_scheduler():
    """
    Instantiates and returns a DDIM Inverse scheduler.
    """
    return DDIMInverseScheduler(
        beta_end=0.012,
        beta_schedule='scaled_linear',
        beta_start=0.00085,
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=1,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=False), 'dddim_inverse'
