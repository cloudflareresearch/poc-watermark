# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from PIL import Image
from torchvision import transforms
from torch import FloatTensor

import numpy as np
import torch
import logging.config
import os
import copy

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(root_path, 'logging.conf')

logging.config.fileConfig(config_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@dataclass
class WatermarkingStableDiffusionPipelineOutput(StableDiffusionPipelineOutput):
    images: Union[List[Image.Image], np.ndarray]
    init_latents: FloatTensor
    nsfw_content_detected: Optional[List[bool]]


class WatermarkingStableDiffusionPipeline(StableDiffusionPipeline):
    """
     Pipeline for text-to-image generation with AI watermarking using Stable Diffusion.

     This subclass is implemented using the Huggingface stable diffusion pipeline source code as a guide. We are
     implementing a custom pipeline so that we can return initial latents used to generated images so that we evaluate
     the efficacy the technique used for generating watermarked images using PRCs as implemented in
     this paper: https://arxiv.org/pdf/2410.07369

     Args:
        vae:
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder:
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer:
            Tokenizes text.
        unet:
            Model to denoise the encoded image latents.
        scheduler:
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker:
            Classification module that estimates whether generated images could be considered offensive or harmful.
        feature_extractor:
            Extract features from generated images; used as inputs to the `safety_checker`.

    """
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae=vae,
                         text_encoder=text_encoder,
                         tokenizer=tokenizer,
                         unet=unet,
                         scheduler=scheduler,
                         safety_checker=safety_checker,
                         feature_extractor=feature_extractor,
                         requires_safety_checker=requires_safety_checker)

        # register needed classes and modules for the models used in this pipeline
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def decode_latents(self, latents):
        """
        Decodes a tensor into a numpy array.

        This is copied over from the source where it is being deprecated. This conversion workers better for our
        purpose as it does not change the dimensions.
        """
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def convert_pil_to_latents(self, img: Image):
        """
        Convert PIL image to numpy array.
        """

        # get the device configured from the pipe configuration
        device = self._execution_device

        with torch.no_grad():
            transform = transforms.ToTensor()
            latent = self.vae.encode(transform(img).unsqueeze(0).to(device) * 2 - 1)
        input_image_latents = 0.18215 * latent.latent_dist.sample()

        return input_image_latents

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        This is the call function to the pipeline for image generation. Note, we have omitted support for callbacks.

         Args:
            prompt:
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height:
                The height in pixels of the generated image.
            width:
                The width in pixels of the generated image.
            num_inference_steps:
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale:
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            num_images_per_prompt:
                The number of images to generate per prompt.
            eta:
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator:
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents:
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type:
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict:
                Whether to return a WatermarkingStableDiffusionPipelineOutput instead of a
                plain tuple.
        """

        # --- Step 1: set height and weight dimensions to what Unet expects
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # --- Step 2: check parameter values are valid (i.e. required values are not None etc)
        self.check_inputs(prompt, height, width, callback_steps=1)

        # --- Step 3: set call parameters
        batch_size = 1 if prompt is not None and isinstance(prompt, str) else len(prompt)

        # Here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # get the device configured from the pipe configuration
        device = self._execution_device

        # --- Step 4: encode input prompt (encoded for the unet hidden states)
        prompt_embeddings, negative_prompt_embeddings = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance)

        # (comments from base class' source code)
        # For classifier free guidance, we need to do two forward passes. Here we concatenate the unconditional and
        #   text embeddings into a single batch to avoid doing two forward passes.
        if do_classifier_free_guidance:
            prompt_embeddings = torch.cat([negative_prompt_embeddings, prompt_embeddings])

        # --- Step 5: set the timesteps in the scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # if we have extra step kwargs, prepare them
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # --- Step 6: prepare the latents (generates a random latents tensor or uses the custom one passed to it)
        # note, a custom generated latent tensor will be passed in when we want to apply a watermark.
        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeddings.dtype,
            device,
            generator,
            latents)

        # copy so we can save these for analysis later
        init_latents = copy.deepcopy(latents)

        # --- Step 7: denoising loop aka where the magic aka generate images
        def do_guidance_predicted_noise_residual(predicted_noise_residual):
            noise_pred_unconditioned, noise_pred_text = predicted_noise_residual.chunk(2)
            return noise_pred_unconditioned + guidance_scale * (noise_pred_text - noise_pred_unconditioned)

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, curr_timestep in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance (i.e. guidance_scale > 1.0)
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, curr_timestep)

                # predict the noise residual
                noise_predication = self.unet(
                    latent_model_input,
                    curr_timestep,
                    encoder_hidden_states=prompt_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )
                noise_predication = noise_predication[0]

                # perform guidance on predicted noise residual
                if do_classifier_free_guidance:
                    noise_predication = do_guidance_predicted_noise_residual(noise_predication)

                # compute the previous noisy sample x_t -> x_t-1 aka denoise the current predicted image latent
                latents = self.scheduler.step(noise_predication, curr_timestep, latents,
                                              **extra_step_kwargs, return_dict=False)
                # note, this is the previous sample (moving back in time so to speak)
                latents = latents[0]

                progress_bar.update()

        if not output_type == "latent":
            # decode the latents to get out generated image(s)
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # check for nsfw output
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # post process the image output from tensor to the configured output type
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # clean up - offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image, init_latents, has_nsfw_concept

        return WatermarkingStableDiffusionPipelineOutput(images=image, init_latents=init_latents,
                                                         nsfw_content_detected=has_nsfw_concept)
