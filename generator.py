# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from PIL import Image
from stable_diffusion.watermark import load_key, generate_watermark, verify_watermark
from stable_diffusion.utils import (get_inference_parameter_values, build_stable_diffusion_pipeline,
                                    convert_pil_to_latents)
from stable_diffusion.schedulers import ddim_scheduler, ddim_inverse_scheduler

import warnings
import argparse
import torch
import logging.config
import os

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

model_cache_path = 'model_cache/'
if not os.path.exists(model_cache_path):
    os.makedirs(model_cache_path)

inference_output_path = 'output/'
if not os.path.exists(inference_output_path):
    os.makedirs(inference_output_path)


def save_generated_image(run_id: str, image: Image.Image):
    """
    Saves generated image.
    """

    output_path = os.path.join(inference_output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_name = f'{run_id}.png'
    generated_image_path = os.path.join(output_path, image_name)
    logger.info(f'Saving watermarked generated image: {generated_image_path}')
    image.save(generated_image_path)


def generate_watermarked_image(prompt: str, params: dict, debug: bool = False, seed: int = 0):
    """
    Generate watermarked image.
    """

    run_id = params.get('run_id')
    seed_value = params.get('seed_value') if seed <= 0 else seed
    height = params.get('height')
    width = params.get('width')
    num_inference_steps = params.get('num_inference_steps')
    model_id = params.get('model_id')

    # --- Build the Stable Diffusion pipeline
    scheduler, scheduler_name = ddim_scheduler()
    pipe, device = build_stable_diffusion_pipeline(model_id, model_cache_path, scheduler, debug=debug)

    if device == 'mps':
        """
        Performance recommendations for Apple M1/M2:
        Ref: https://huggingface.co/docs/diffusers/v0.21.0/en/optimization/mps#performance-recommendations
        """
        pipe.enable_attention_slicing()

    # --- Generate initial latents containing a watermark

    # Load the watermarking key from disk.
    watermarking_key_path = "test_key.pem"
    watermarking_key = load_key(watermarking_key_path)

    # Generate the watermarked initial latent
    watermarked_init_latent = generate_watermark(watermarking_key)

    # --- Generate image
    """
    For a reproducible pipeline on this machine, Huggingface recommends passing a CPU Generator. Calling
    torch.manual_seed() automatically creates a CPU Generator that can be passed to the pipeline even if the pipeline
    is set to run on a GPU run on a GPU. Ref: https://huggingface.co/docs/diffusers/en/using-diffusers/reusing_seeds

    The seed value used to generate the initial random latents in the generation pipeline.
    """
    generator = torch.manual_seed(seed_value)

    logger.info(f'Generating image for prompt: "{prompt}"')
    generated_image, _ = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps,
                              generator=generator, latents=watermarked_init_latent, return_dict=False)
    generated_image = generated_image[0]

    # --- Make sure the latent is verifiable
    logger.info(f'Verifying watermark...')

    scheduler, _ = ddim_inverse_scheduler()
    pipe.scheduler = scheduler

    # Convert input image to latent space
    clean_image_latent = convert_pil_to_latents(pipe, generated_image)

    # Invert the sampling process that generated the original image using that image converted to
    # the latent space.
    inverted_latent, _ = pipe("", output_type='latent', latents=clean_image_latent, height=height, width=width,
                               num_inference_steps=num_inference_steps, return_dict=False)

    # Check if the inverted latent has a valid codeword
    if not verify_watermark(watermarking_key, inverted_latent):
        # This might happen inverted_latent is a bad approximation of init_latent, in particular if
        # too many signs don't match. See the evaluator module for details.
        raise ValueError('Watermark not verified')

    # --- Save generated image
    logger.info(f'Saving image...')
    save_generated_image(run_id, generated_image)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--prompt', type=str,
                        help='Prompt to generate a single image.',
                        required=True)
    parser.add_argument('--debug', type=bool,
                        help='Puts this script in debug mode, which sets the device target to CPU.',
                        default=False, required=False)
    parser.add_argument('--seed_value', type=int,
                        help='Random seed value used to generate images.',
                        default=8, required=False)
    args = parser.parse_args()

    """
    As of Feb 2025 there is a bug with Apple M1/M2 chips when running stable diffusion pipelines in Pycharm in
    debug mode where `mps` is the targeted device it blows up and yields a IOGPUMetalCommandBuffer -
    'commit an already committed command buffer' assertion error.

    If you are debugging this on a machine with Apple M1/M2 chips in PyCharm, set the debug flag to True. This will
    set the stable diffusion pipeline to use the CPU.
    """
    is_debug_mode = args.debug
    user_prompt = args.prompt
    seed_value = args.seed_value

    # --- Get default inference parameter values
    default_inference_params = get_inference_parameter_values()

    generate_watermarked_image(user_prompt, default_inference_params, debug=is_debug_mode, seed=seed_value)
