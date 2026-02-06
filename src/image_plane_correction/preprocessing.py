import jax
import jax.numpy as jnp
from jaxtyping import Array

from .util import rescale_quantile, gaussian_filter, circular_mask

def normalize(image: Array):
    """
    Rescales image pixels to reduce high and low outliers. Additionally,
    all output pixels are scaled to the range [0.0, 1.0].
    """
    return rescale_quantile(image, 0.01, 0.99)
    
def normalize_high(image: Array):
    """
    Rescales image pixels to retain the upper 50% to 99% percentile of
    bright pixels. Additionally, all output pixels are scaled to the
    range [0.0, 1.0].
    """
    return rescale_quantile(image, 0.50, 0.99)

@jax.jit
def horizon_mask(image: Array, r = 0.7, sigma=30.0):
    """
    Smoothly masks out the horizon of an image. Is done to reduce horizon
    effects and RFI near the horizon, while the smooth transition helps
    to reduce detrimental effects on the optical flow model.
    """
    # use FFT-based convolution for better performance
    mask = gaussian_filter(circular_mask(r=r), sigma=sigma, method="fft")
    return image * mask


def preprocess(image: Array, sky: Array, weight=1.0):
    """
    The standard preprocessing pipeline to match up an observed and theoretical sky.
    Output images are optimized to produce a smooth flow with signals after being
    passed into an optical flow model.
    """
    # mask out horizon RFI/artifacts
    image, sky = horizon_mask(image), horizon_mask(sky)

    # rescale pixel brightnesses to approximately the same level
    image, sky = normalize_high(image), normalize_high(sky)
    
    # a weighting function to help separate bright sources from noise
    image, sky = image ** weight, sky ** weight

    # gaussian blur for smoothness and to help reduce the effect of differeing PSF shapes
    image, sky = gaussian_filter(image, sigma=5.0), gaussian_filter(sky, sigma=5.0)

    # re-normalize image values after blur
    image, sky = normalize(image), normalize(sky)

    return image, sky