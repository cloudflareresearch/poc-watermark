# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from PIL import Image
from stable_diffusion.watermark import load_key, verify_watermark
from stable_diffusion.schedulers import ddim_inverse_scheduler
from stable_diffusion.utils import (get_inference_parameter_values, build_stable_diffusion_pipeline,
                                    convert_pil_to_latents)

import warnings
import argparse
import logging.config
import os

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

model_cache_path = 'model_cache/'
if not os.path.exists(model_cache_path):
    os.makedirs(model_cache_path)


def recover_initial_latent(model: str, img: Image, prompt: str = "", debug: bool = False):
    """
    Computes an approximation of the initial latent used to generate the image.
    """

    scheduler, _ = ddim_inverse_scheduler()
    pipe, device = build_stable_diffusion_pipeline(model, model_cache_path, scheduler, debug=debug)

    if device == 'mps':
        """
        Performance recommendations for Apple M1/M2:
        Ref: https://huggingface.co/docs/diffusers/v0.21.0/en/optimization/mps#performance-recommendations
        """
        pipe.enable_attention_slicing()

    # Convert the input image to latent space
    input_image_latent = convert_pil_to_latents(pipe, img)

    # Invert the sampling process that possibly generated the input image
    inverted_latent, _ = pipe(prompt, output_type='latent', latents=input_image_latent, height=height, width=width,
                              num_inference_steps=num_inference_steps, return_dict=False)

    return inverted_latent


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--image_path', type=str,
                        help='Fully qualified path of image path.',
                        required=True)
    parser.add_argument('--debug', type=bool,
                        help='Puts this script in debug mode, which sets the device target to CPU.',
                        default=False, required=False)
    args = parser.parse_args()

    # --- Get default inference parameter values
    params = get_inference_parameter_values()
    height = params.get('height')
    width = params.get('width')
    num_inference_steps = params.get('num_inference_steps')
    model = params.get('model_id')

    # --- Load input image and check that the image dimensions are valid
    image_path = args.image_path
    input_image = None

    if os.path.exists(image_path):
        input_image = Image.open(image_path)

        if not input_image.width == width and not input_image.height == height:
            raise ValueError('Image dimensions must be {}x{}. Please update image.'.format(width, height))
    else:
        raise ValueError('File does not exist. Please update the --image_path.')

    """
    As of Feb 2025 there is a bug with Apple M1/M2 chips when running stable diffusion pipelines in Pycharm in
    debug mode. When `mps` is the targeted device, it blows up and yields a IOGPUMetalCommandBuffer -
    `commit an already committed command buffer' assertion error.

    If you are debugging this on a machine with Apple M1/M2 chips in PyCharm, set the debug flag to True. This will
    set the stable diffusion pipeline to use the CPU.
    """
    is_debug_mode = args.debug

    # --- Get the approximate initial latent that was possibly used to generate the input image
    logger.info('Computing the inverted latent')
    inverted_latent = recover_initial_latent(model, input_image, debug=is_debug_mode)

    # --- Check for watermark
    logger.info('Checking the initial latent for the watermark')

    # Load the encoding key and decoding key from disk
    watermarking_key_path = "test_key.pem"
    watermarking_key = load_key(watermarking_key_path)

    # Check if the inverted latent has a valid codeword
    watermark_verified = verify_watermark(watermarking_key, inverted_latent)
    if watermark_verified:
        logger.info('Watermark verified.')
    else:
        logger.exception('Watermark NOT verified.')
