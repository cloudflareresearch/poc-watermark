# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from torch import FloatTensor

import torch
import numpy as np
import functools
from prc import *

LATENTS_SHAPE = [1, 4, 64, 64]
LATENTS_LEN = functools.reduce(int.__mul__, LATENTS_SHAPE)

def load_key(path: str):
    """
    Loads the watermarking key from disk.
    """

    with open(path) as f:
        pem_str = f.read()
        key = key_from_pem(pem_str)

    return key


def generate_watermark(key):
    """
    Generates a random latent tensor with a LDPC encoded watermark for use in a Stable Diffusion
    Variable Autoencoder (VAE) model.
    """

    # Generate a normally distributed latent. For the model's the default image
    # size of 512x512 bits, `LATENTS_SHAPE = [1, 4, 64, 64]`.
    initial_latent = np.abs(np.random.randn(*LATENTS_SHAPE))

    with np.nditer(initial_latent, op_flags=['readwrite']) as it:
            codeword = encode(key)
            assert len(codeword) == LATENTS_LEN
            for (i, x) in enumerate(it):
                 # `codeword[i]` is a `bool` representing the `i`-th bit of
                 # the codeword.
                 x *= 1 if codeword[i] else -1

    watermarked_latent = torch.from_numpy(initial_latent).to(dtype=torch.float32)

    return watermarked_latent


def verify_watermark(key, inverted_latent: FloatTensor):
    """
    Detects the watermark by checking if the inverted latent tensor is a valid codeword.
    """

    watermark_verified = False
    with np.nditer(inverted_latent.cpu().numpy()) as it:
        inverted_codeword = []
        for x in it:
             inverted_codeword.append(x > 0)

        if detect(key, inverted_codeword):
             watermark_verified = True

    return watermark_verified
