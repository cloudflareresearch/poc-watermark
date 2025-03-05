# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from PIL import Image
from stable_diffusion.utils import get_inference_parameter_values
from stable_diffusion.schedulers import ddim_scheduler, ddim_inverse_scheduler
from stable_diffusion.custom_pipeline import WatermarkingStableDiffusionPipeline
from stable_diffusion.utils import get_device
from torch import FloatTensor

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


def build_custom_stable_diffusion_pipeline(model: str, cache_path: str, scheduler, debug: bool = False):
    """
    Returns a custom stable diffusion pipeline with the detected device set.

    This pipeline returns the initial latent used to generate the image. This is needed for evaluator analysis.
    """

    device = get_device()

    pipe = WatermarkingStableDiffusionPipeline.from_pretrained(
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


def compare_latent_tensors(init_latent_tensor: FloatTensor, inverted_latent_tensor: FloatTensor):
    """
    Compares the signs of the initial latent tensor to the inverted_latent tensor.
    """

    # Remove batch dimension. The batch dimension size should be 1 since we only generate a single image in the
    # generator per run.
    y = init_latent_tensor[-1]
    w = inverted_latent_tensor[-1]

    # Compute signs tensors
    init_latent_tensor_signs = torch.sign(y)
    inverted_latent_tensor_signs = torch.sign(w)

    # Do element-wise comparison of signs tensors
    signs_compare_tensor = torch.eq(init_latent_tensor_signs, inverted_latent_tensor_signs)

    # get total element count, (both tensors are the same shape)
    tensor_element_count = torch.numel(y)

    # count all the "True" elements (function treats True as 1)
    matching_signs_count = signs_compare_tensor.sum().item()
    matching_signs_percent = (matching_signs_count / tensor_element_count)

    return {
        'tensor_element_count': tensor_element_count,
        'matching_signs_count': matching_signs_count,
        'matching_signs_percent': matching_signs_percent
    }


def generate_image(prompt: str, params: dict, debug: bool = False, seed: int = 0):
    """
    Generates an unwatermarked image.
    """

    seed_value = params.get('seed_value') if seed <= 0 else seed
    height = params.get('height')
    width = params.get('width')
    num_inference_steps = params.get('num_inference_steps')
    model_id = params.get('model_id')

    # --- Build the Stable Diffusion pipeline
    scheduler, scheduler_name = ddim_scheduler()
    pipe, device = build_custom_stable_diffusion_pipeline(model_id, model_cache_path, scheduler, debug=debug)

    if device == 'mps':
        """
        Performance recommendations for Apple M1/M2:
        Ref: https://huggingface.co/docs/diffusers/v0.21.0/en/optimization/mps#performance-recommendations
        """
        pipe.enable_attention_slicing()

    # --- Generate the image
    """
    For a reproducible pipeline on this machine, Huggingface recommends passing a CPU Generator. Calling
    torch.manual_seed() automatically creates a CPU Generator that can be passed to the pipeline even if the pipeline
    is set to run on a GPU run on a GPU. Ref: https://huggingface.co/docs/diffusers/en/using-diffusers/reusing_seeds

    The seed value used to generate the initial random latent in the generation pipeline.
    """
    generator = torch.manual_seed(seed_value)

    logger.info(f'Generating image for prompt: "{prompt}"')
    image, init_latent, _ = pipe(prompt, height=height, width=width,
                                 num_inference_steps=num_inference_steps,
                                 generator=generator, return_dict=False)
    image = image[0]

    # --- Compute inverse latent for the generated image + prompt
    logger.info(f'Generating inverse latent for generated image...')

    scheduler, _ = ddim_inverse_scheduler()
    pipe.scheduler = scheduler

    # Convert input image to latent space
    generated_image_latent = pipe.convert_pil_to_latents(image)

    # Invert the sampling process that generated the original image using that image converted to
    #   the latent space.
    inverted_latent, _, _ = pipe("", output_type='latent', latents=generated_image_latent, height=height, width=width,
                                  num_inference_steps=num_inference_steps, return_dict=False)

    return image, init_latent, inverted_latent


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

    # Get default inference parameter values
    default_inference_params = get_inference_parameter_values()

    # Generate unwatermarked image
    generated_image, init_latent, inverted_latent = generate_image(user_prompt, default_inference_params,
                                                                   debug=is_debug_mode, seed=seed_value)

    # Compare similarity of init latent used generate the image and the inverted latent which is an
    # approximation of the init latent.
    results = compare_latent_tensors(init_latent, inverted_latent)

    element_count = results['tensor_element_count']
    matching_signs_count = results['matching_signs_count']
    matching_signs_percent = results['matching_signs_percent']

    logger.info(f'Total elements in latent tensors: {element_count}')
    logger.info(f'Total matching signs (Y vs W): {matching_signs_count} (~{matching_signs_percent:.0%}))')
