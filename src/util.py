"""Utility functions (mostly ported from other libraries to JAX)"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike


def indices(m, n):
    return jnp.indices((m, n)).transpose(1, 2, 0)

def horizon_mask(N=4096, r=0.85):
    idxs = indices(N, N) - N/2
    return jnp.sqrt(idxs[:, :, 0]**2.0 + idxs[:, :, 1]**2.0) < N * r * 0.5

# https://github.com/jax-ml/jax/pull/26011/files
import functools
from scipy.ndimage import _ni_support
def _gaussian(x, sigma):
    return jnp.exp(-0.5 / sigma ** 2 * x ** 2) / jnp.sqrt(2 * jnp.pi * sigma ** 2)

def _grad_order(func, order):
    """Compute higher order grads recursively"""
    if order == 0:
      return func

    return jax.grad(_grad_order(func, order - 1))


def _gaussian_kernel1d(sigma, order, radius):
    """Computes a 1-D Gaussian convolution kernel"""
    if order < 0:
        raise ValueError(f'Order must be non-negative, got {order}')

    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    func = _grad_order(functools.partial(_gaussian, sigma=sigma), order)
    kernel = jax.vmap(func)(x)

    if order == 0:
       return kernel / jnp.sum(kernel)

    return kernel

def gaussian_filter1d(
      input: ArrayLike,
      sigma: float,
      axis: int = -1,
      order: int = 0,
      mode: str = 'reflect',
      cval: float = 0.0,
      truncate: float = 4.0,
      *,
      radius: int | None = None,
      method: str = "auto"):
    """Compute a 1D Gaussian filter on the input array along the specified axis.
    Args:
        input: N-dimensional input array to filter.
        sigma: The standard deviation of the Gaussian filter.
        axis: The axis along which to apply the filter.
        order: The order of the Gaussian filter.
        mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
        cval: The value to use for padding the input array.
        truncate: The number of standard deviations to include in the filter.
        radius: The radius of the filter. Overrides `truncate` if provided.
        method: The method to use for the convolution.
    Returns:
        The filtered array.
    Examples:
        >>> from jax import numpy as jnp
        >>> import jax
        >>> input = jnp.arange(12.0).reshape(3, 4)
        >>> input
        Array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.]], dtype=float32)
        >>> jax.scipy.ndimage.gaussian_filter1d(input, sigma=1.0, axis=0, order=0)
       Array([[2.8350844, 3.8350847, 4.8350844, 5.8350844],
              [4.0000005, 5.       , 6.       , 7.0000005],
              [5.1649156, 6.1649156, 7.164916 , 8.164916 ]], dtype=float32)
    """
    if radius is None:
        radius = int(truncate * sigma + 0.5)

    if radius < 0:
        raise ValueError(f'Radius must be non-negative, got {radius}')

    if sigma <= 0:
        raise ValueError(f'Sigma must be positive, got {sigma}')

    pad_width = [(0, 0)] * input.ndim
    pad_width[axis] = (int(radius), int(radius))

    pad_kwargs = {'mode': mode}

    if mode == 'constant':
       # jnp.pad errors if constant_values is provided and mode is not 'constant'
       pad_kwargs['constant_values'] = cval

    input_pad = jnp.pad(input, pad_width=pad_width, **pad_kwargs)

    kernel = _gaussian_kernel1d(sigma, order=order, radius=radius)

    axes = list(range(input.ndim))
    axes.pop(axis)
    kernel = jnp.expand_dims(kernel, axes)

    # boundary handling is done by jnp.pad, so we use the fixed valid mode
    return jax.scipy.signal.convolve(input_pad, kernel, mode="valid", method=method)

from collections.abc import Sequence

def gaussian_filter(
      input: ArrayLike,
      sigma: float | Sequence[float],
      order: int | Sequence[int] = 0,
      mode: str = 'reflect',
      cval: float | Sequence[float] = 0.0,
      truncate: float | Sequence[float] = 4.0,
      *,
      radius: None | Sequence[int] = None,
      axes: Sequence[int] = None,
      method="auto",
    ):
    """Gaussian filter for N-dimensional input
    
     Args:
        input: N-dimensional input array to filter.
        sigma: The standard deviation of the Gaussian filter.
        order: The order of the Gaussian filter.
        mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
        cval: The value to use for padding the input array.
        truncate: The number of standard deviations to include in the filter.
        radius: The radius of the filter. Overrides `truncate` if provided.
        method: The method to use for the convolution.
    """
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radii = _ni_support._normalize_sequence(radius, num_axes)

    # the loop goes over the input axes, so it is always low-dimensional and
    # keeping a Python loop is ok
    for idx in range(input.ndim):
       input = gaussian_filter1d(
          input,
          sigmas[idx],
          axis=idx,
          order=orders[idx],
          mode=modes[idx],
          cval=cval,
          truncate=truncate,
          radius=radii[idx],
          method=method,
       )

    return input

@jax.jit
def rescale_quantile(image, a, b):
    qa = jnp.quantile(image, a)
    qb = jnp.quantile(image, b)
    return jnp.clip(
        (image - qa)
        / (qb - qa),
        0,
        1,
    )


def clip_quantile(image, a, b):
    return jnp.clip(image, jnp.quantile(image, a), jnp.quantile(image, b))


# https://stackoverflow.com/a/43346070
def gkern(l=5, sig=1.0):
    """\
    Creates gaussian kernel with side length `l` and a sigma of `sig`.
    Reaches a maximum of 1 at its center value
    """
    ax = jnp.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(sig))
    kernel = jnp.outer(gauss, gauss)
    return kernel / kernel.max()


# ported to jax from from https://github.com/matplotlib/matplotlib/blob/v3.9.2/lib/matplotlib/colors.py#L2235
@jax.jit
def hsv_to_rgb(hsv: Array):
    """
    Convert HSV values to RGB.

    Parameters
    ----------
    hsv : (..., 3) `jax.Array`
       All values assumed to be in range [0, 1]

    Returns
    -------
    (..., 3) `jax.Array`
       Colors converted to RGB values in range [0, 1]
    """
    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError(
            "Last dimension of input array must be 3; " f"shape {hsv.shape} was found."
        )

    in_shape = hsv.shape

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = jnp.zeros_like(h)
    g = jnp.zeros_like(h)
    b = jnp.zeros_like(h)

    i = (h * 6.0).astype(jnp.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = r + jnp.where(i % 6 == 0, v, 0)
    g = g + jnp.where(i % 6 == 0, t, 0)
    b = b + jnp.where(i % 6 == 0, p, 0)

    r = r + jnp.where(i % 6 == 1, q, 0)
    g = g + jnp.where(i % 6 == 1, v, 0)
    b = b + jnp.where(i % 6 == 1, p, 0)

    r = r + jnp.where(i % 6 == 2, p, 0)
    g = g + jnp.where(i % 6 == 2, v, 0)
    b = b + jnp.where(i % 6 == 2, t, 0)

    r = r + jnp.where(i % 6 == 3, p, 0)
    g = g + jnp.where(i % 6 == 3, q, 0)
    b = b + jnp.where(i % 6 == 3, v, 0)

    r = r + jnp.where(i % 6 == 4, t, 0)
    g = g + jnp.where(i % 6 == 4, p, 0)
    b = b + jnp.where(i % 6 == 4, v, 0)

    r = r + jnp.where(i % 6 == 5, v, 0)
    g = g + jnp.where(i % 6 == 5, p, 0)
    b = b + jnp.where(i % 6 == 5, q, 0)

    r = r + jnp.where(i % 6 == 6, v, 0)
    g = g + jnp.where(i % 6 == 6, v, 0)
    b = b + jnp.where(i % 6 == 6, v, 0)

    rgb = jnp.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)


# porting match_histograms to jax
# https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/exposure/histogram_matching.py#L33-L93
def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    if source.dtype.kind == "u":
        src_lookup = source.reshape(-1)
        src_counts = jnp.bincount(src_lookup)
        tmpl_counts = jnp.bincount(template.reshape(-1))

        # omit values where the count was 0
        tmpl_values = jnp.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = jnp.unique(
            source.reshape(-1), return_inverse=True, return_counts=True
        )
        tmpl_values, tmpl_counts = jnp.unique(template.reshape(-1), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = jnp.cumsum(src_counts) / source.size
    tmpl_quantiles = jnp.cumsum(tmpl_counts) / template.size

    interp_a_values = jnp.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(source.shape)


def match_histograms(image, reference):
    """Adjust an image so that its cumulative histogram matches that of another.

    We assume the image only has one color channel (e.g. is greyscale).

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/

    """
    if image.ndim != reference.ndim:
        raise ValueError(
            "Image and reference must have the same number " "of channels."
        )

    # _match_cumulative_cdf will always return float64 due to np.interp
    matched = _match_cumulative_cdf(image, reference)

    return matched
