# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision import transforms

import random
import logging.config
import os
import uuid
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(root_path, 'logging.conf')

logging.config.fileConfig(config_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def get_device():
    """
    Returns the detected device type. If no GPU is available, it returns CPU.
    """
    if torch.backends.mps.is_available():
        logger.info("MPS is available.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("CUDA is available.")
        device = torch.device("cuda")
    else:
        logger.info("No GPUs found. Use CPU.")
        device = torch.device("cpu")

    return device


def generate_random_seed():
    return random.randint(0, 2 ** 32 - 1)


def get_inference_parameter_values():
    """
    Returns the inference hyperparameter values for the PoC.
    """

    run_id = str(uuid.uuid4().hex)
    seed_value = 8
    height = 512
    width = 512
    num_inference_steps = 50
    model_id = 'stabilityai/stable-diffusion-2-1-base'

    return {
        'run_id': run_id,
        'seed_value': seed_value,
        'height': height,
        'width': width,
        'num_inference_steps': num_inference_steps,
        'model_id': model_id}


def build_stable_diffusion_pipeline(model: str, cache_path: str, scheduler, debug: bool = False):
    """
    Returns a stable diffusion pipeline with the detected device set.
    """

    device = get_device()

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        cache_dir=cache_path,
    )

    if debug:
        pipe = pipe.to('cpu')
    else:
        pipe = pipe.to(device)

    return pipe, device


def convert_pil_to_latents(pipe: StableDiffusionPipeline, img: Image):
    """
    Convert PIL image to numpy array.
    """

    # get the device configured from the pipe configuration
    device = pipe.device

    with torch.no_grad():
        transform = transforms.ToTensor()
        latent = pipe.vae.encode(transform(img).unsqueeze(0).to(device) * 2 - 1)
    input_image_latents = 0.18215 * latent.latent_dist.sample()

    return input_image_latents
