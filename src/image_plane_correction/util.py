"""Utility functions (mostly ported from other libraries to JAX)"""

import os
import re
from collections.abc import Iterable, Sequence
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike


def indices(m, n):
    return jnp.indices((m, n)).transpose(1, 2, 0)

def circular_mask(N=4096, r=0.85):
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
    
@jax.jit
def rescale_absolute(image, qa, qb):
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

    hsv = jnp.clip(hsv, 0.0, 1.0)

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

_viridis_data = [[0.267004, 0.004874, 0.329415],
                 [0.268510, 0.009605, 0.335427],
                 [0.269944, 0.014625, 0.341379],
                 [0.271305, 0.019942, 0.347269],
                 [0.272594, 0.025563, 0.353093],
                 [0.273809, 0.031497, 0.358853],
                 [0.274952, 0.037752, 0.364543],
                 [0.276022, 0.044167, 0.370164],
                 [0.277018, 0.050344, 0.375715],
                 [0.277941, 0.056324, 0.381191],
                 [0.278791, 0.062145, 0.386592],
                 [0.279566, 0.067836, 0.391917],
                 [0.280267, 0.073417, 0.397163],
                 [0.280894, 0.078907, 0.402329],
                 [0.281446, 0.084320, 0.407414],
                 [0.281924, 0.089666, 0.412415],
                 [0.282327, 0.094955, 0.417331],
                 [0.282656, 0.100196, 0.422160],
                 [0.282910, 0.105393, 0.426902],
                 [0.283091, 0.110553, 0.431554],
                 [0.283197, 0.115680, 0.436115],
                 [0.283229, 0.120777, 0.440584],
                 [0.283187, 0.125848, 0.444960],
                 [0.283072, 0.130895, 0.449241],
                 [0.282884, 0.135920, 0.453427],
                 [0.282623, 0.140926, 0.457517],
                 [0.282290, 0.145912, 0.461510],
                 [0.281887, 0.150881, 0.465405],
                 [0.281412, 0.155834, 0.469201],
                 [0.280868, 0.160771, 0.472899],
                 [0.280255, 0.165693, 0.476498],
                 [0.279574, 0.170599, 0.479997],
                 [0.278826, 0.175490, 0.483397],
                 [0.278012, 0.180367, 0.486697],
                 [0.277134, 0.185228, 0.489898],
                 [0.276194, 0.190074, 0.493001],
                 [0.275191, 0.194905, 0.496005],
                 [0.274128, 0.199721, 0.498911],
                 [0.273006, 0.204520, 0.501721],
                 [0.271828, 0.209303, 0.504434],
                 [0.270595, 0.214069, 0.507052],
                 [0.269308, 0.218818, 0.509577],
                 [0.267968, 0.223549, 0.512008],
                 [0.266580, 0.228262, 0.514349],
                 [0.265145, 0.232956, 0.516599],
                 [0.263663, 0.237631, 0.518762],
                 [0.262138, 0.242286, 0.520837],
                 [0.260571, 0.246922, 0.522828],
                 [0.258965, 0.251537, 0.524736],
                 [0.257322, 0.256130, 0.526563],
                 [0.255645, 0.260703, 0.528312],
                 [0.253935, 0.265254, 0.529983],
                 [0.252194, 0.269783, 0.531579],
                 [0.250425, 0.274290, 0.533103],
                 [0.248629, 0.278775, 0.534556],
                 [0.246811, 0.283237, 0.535941],
                 [0.244972, 0.287675, 0.537260],
                 [0.243113, 0.292092, 0.538516],
                 [0.241237, 0.296485, 0.539709],
                 [0.239346, 0.300855, 0.540844],
                 [0.237441, 0.305202, 0.541921],
                 [0.235526, 0.309527, 0.542944],
                 [0.233603, 0.313828, 0.543914],
                 [0.231674, 0.318106, 0.544834],
                 [0.229739, 0.322361, 0.545706],
                 [0.227802, 0.326594, 0.546532],
                 [0.225863, 0.330805, 0.547314],
                 [0.223925, 0.334994, 0.548053],
                 [0.221989, 0.339161, 0.548752],
                 [0.220057, 0.343307, 0.549413],
                 [0.218130, 0.347432, 0.550038],
                 [0.216210, 0.351535, 0.550627],
                 [0.214298, 0.355619, 0.551184],
                 [0.212395, 0.359683, 0.551710],
                 [0.210503, 0.363727, 0.552206],
                 [0.208623, 0.367752, 0.552675],
                 [0.206756, 0.371758, 0.553117],
                 [0.204903, 0.375746, 0.553533],
                 [0.203063, 0.379716, 0.553925],
                 [0.201239, 0.383670, 0.554294],
                 [0.199430, 0.387607, 0.554642],
                 [0.197636, 0.391528, 0.554969],
                 [0.195860, 0.395433, 0.555276],
                 [0.194100, 0.399323, 0.555565],
                 [0.192357, 0.403199, 0.555836],
                 [0.190631, 0.407061, 0.556089],
                 [0.188923, 0.410910, 0.556326],
                 [0.187231, 0.414746, 0.556547],
                 [0.185556, 0.418570, 0.556753],
                 [0.183898, 0.422383, 0.556944],
                 [0.182256, 0.426184, 0.557120],
                 [0.180629, 0.429975, 0.557282],
                 [0.179019, 0.433756, 0.557430],
                 [0.177423, 0.437527, 0.557565],
                 [0.175841, 0.441290, 0.557685],
                 [0.174274, 0.445044, 0.557792],
                 [0.172719, 0.448791, 0.557885],
                 [0.171176, 0.452530, 0.557965],
                 [0.169646, 0.456262, 0.558030],
                 [0.168126, 0.459988, 0.558082],
                 [0.166617, 0.463708, 0.558119],
                 [0.165117, 0.467423, 0.558141],
                 [0.163625, 0.471133, 0.558148],
                 [0.162142, 0.474838, 0.558140],
                 [0.160665, 0.478540, 0.558115],
                 [0.159194, 0.482237, 0.558073],
                 [0.157729, 0.485932, 0.558013],
                 [0.156270, 0.489624, 0.557936],
                 [0.154815, 0.493313, 0.557840],
                 [0.153364, 0.497000, 0.557724],
                 [0.151918, 0.500685, 0.557587],
                 [0.150476, 0.504369, 0.557430],
                 [0.149039, 0.508051, 0.557250],
                 [0.147607, 0.511733, 0.557049],
                 [0.146180, 0.515413, 0.556823],
                 [0.144759, 0.519093, 0.556572],
                 [0.143343, 0.522773, 0.556295],
                 [0.141935, 0.526453, 0.555991],
                 [0.140536, 0.530132, 0.555659],
                 [0.139147, 0.533812, 0.555298],
                 [0.137770, 0.537492, 0.554906],
                 [0.136408, 0.541173, 0.554483],
                 [0.135066, 0.544853, 0.554029],
                 [0.133743, 0.548535, 0.553541],
                 [0.132444, 0.552216, 0.553018],
                 [0.131172, 0.555899, 0.552459],
                 [0.129933, 0.559582, 0.551864],
                 [0.128729, 0.563265, 0.551229],
                 [0.127568, 0.566949, 0.550556],
                 [0.126453, 0.570633, 0.549841],
                 [0.125394, 0.574318, 0.549086],
                 [0.124395, 0.578002, 0.548287],
                 [0.123463, 0.581687, 0.547445],
                 [0.122606, 0.585371, 0.546557],
                 [0.121831, 0.589055, 0.545623],
                 [0.121148, 0.592739, 0.544641],
                 [0.120565, 0.596422, 0.543611],
                 [0.120092, 0.600104, 0.542530],
                 [0.119738, 0.603785, 0.541400],
                 [0.119512, 0.607464, 0.540218],
                 [0.119423, 0.611141, 0.538982],
                 [0.119483, 0.614817, 0.537692],
                 [0.119699, 0.618490, 0.536347],
                 [0.120081, 0.622161, 0.534946],
                 [0.120638, 0.625828, 0.533488],
                 [0.121380, 0.629492, 0.531973],
                 [0.122312, 0.633153, 0.530398],
                 [0.123444, 0.636809, 0.528763],
                 [0.124780, 0.640461, 0.527068],
                 [0.126326, 0.644107, 0.525311],
                 [0.128087, 0.647749, 0.523491],
                 [0.130067, 0.651384, 0.521608],
                 [0.132268, 0.655014, 0.519661],
                 [0.134692, 0.658636, 0.517649],
                 [0.137339, 0.662252, 0.515571],
                 [0.140210, 0.665859, 0.513427],
                 [0.143303, 0.669459, 0.511215],
                 [0.146616, 0.673050, 0.508936],
                 [0.150148, 0.676631, 0.506589],
                 [0.153894, 0.680203, 0.504172],
                 [0.157851, 0.683765, 0.501686],
                 [0.162016, 0.687316, 0.499129],
                 [0.166383, 0.690856, 0.496502],
                 [0.170948, 0.694384, 0.493803],
                 [0.175707, 0.697900, 0.491033],
                 [0.180653, 0.701402, 0.488189],
                 [0.185783, 0.704891, 0.485273],
                 [0.191090, 0.708366, 0.482284],
                 [0.196571, 0.711827, 0.479221],
                 [0.202219, 0.715272, 0.476084],
                 [0.208030, 0.718701, 0.472873],
                 [0.214000, 0.722114, 0.469588],
                 [0.220124, 0.725509, 0.466226],
                 [0.226397, 0.728888, 0.462789],
                 [0.232815, 0.732247, 0.459277],
                 [0.239374, 0.735588, 0.455688],
                 [0.246070, 0.738910, 0.452024],
                 [0.252899, 0.742211, 0.448284],
                 [0.259857, 0.745492, 0.444467],
                 [0.266941, 0.748751, 0.440573],
                 [0.274149, 0.751988, 0.436601],
                 [0.281477, 0.755203, 0.432552],
                 [0.288921, 0.758394, 0.428426],
                 [0.296479, 0.761561, 0.424223],
                 [0.304148, 0.764704, 0.419943],
                 [0.311925, 0.767822, 0.415586],
                 [0.319809, 0.770914, 0.411152],
                 [0.327796, 0.773980, 0.406640],
                 [0.335885, 0.777018, 0.402049],
                 [0.344074, 0.780029, 0.397381],
                 [0.352360, 0.783011, 0.392636],
                 [0.360741, 0.785964, 0.387814],
                 [0.369214, 0.788888, 0.382914],
                 [0.377779, 0.791781, 0.377939],
                 [0.386433, 0.794644, 0.372886],
                 [0.395174, 0.797475, 0.367757],
                 [0.404001, 0.800275, 0.362552],
                 [0.412913, 0.803041, 0.357269],
                 [0.421908, 0.805774, 0.351910],
                 [0.430983, 0.808473, 0.346476],
                 [0.440137, 0.811138, 0.340967],
                 [0.449368, 0.813768, 0.335384],
                 [0.458674, 0.816363, 0.329727],
                 [0.468053, 0.818921, 0.323998],
                 [0.477504, 0.821444, 0.318195],
                 [0.487026, 0.823929, 0.312321],
                 [0.496615, 0.826376, 0.306377],
                 [0.506271, 0.828786, 0.300362],
                 [0.515992, 0.831158, 0.294279],
                 [0.525776, 0.833491, 0.288127],
                 [0.535621, 0.835785, 0.281908],
                 [0.545524, 0.838039, 0.275626],
                 [0.555484, 0.840254, 0.269281],
                 [0.565498, 0.842430, 0.262877],
                 [0.575563, 0.844566, 0.256415],
                 [0.585678, 0.846661, 0.249897],
                 [0.595839, 0.848717, 0.243329],
                 [0.606045, 0.850733, 0.236712],
                 [0.616293, 0.852709, 0.230052],
                 [0.626579, 0.854645, 0.223353],
                 [0.636902, 0.856542, 0.216620],
                 [0.647257, 0.858400, 0.209861],
                 [0.657642, 0.860219, 0.203082],
                 [0.668054, 0.861999, 0.196293],
                 [0.678489, 0.863742, 0.189503],
                 [0.688944, 0.865448, 0.182725],
                 [0.699415, 0.867117, 0.175971],
                 [0.709898, 0.868751, 0.169257],
                 [0.720391, 0.870350, 0.162603],
                 [0.730889, 0.871916, 0.156029],
                 [0.741388, 0.873449, 0.149561],
                 [0.751884, 0.874951, 0.143228],
                 [0.762373, 0.876424, 0.137064],
                 [0.772852, 0.877868, 0.131109],
                 [0.783315, 0.879285, 0.125405],
                 [0.793760, 0.880678, 0.120005],
                 [0.804182, 0.882046, 0.114965],
                 [0.814576, 0.883393, 0.110347],
                 [0.824940, 0.884720, 0.106217],
                 [0.835270, 0.886029, 0.102646],
                 [0.845561, 0.887322, 0.099702],
                 [0.855810, 0.888601, 0.097452],
                 [0.866013, 0.889868, 0.095953],
                 [0.876168, 0.891125, 0.095250],
                 [0.886271, 0.892374, 0.095374],
                 [0.896320, 0.893616, 0.096335],
                 [0.906311, 0.894855, 0.098125],
                 [0.916242, 0.896091, 0.100717],
                 [0.926106, 0.897330, 0.104071],
                 [0.935904, 0.898570, 0.108131],
                 [0.945636, 0.899815, 0.112838],
                 [0.955300, 0.901065, 0.118128],
                 [0.964894, 0.902323, 0.123941],
                 [0.974417, 0.903590, 0.130215],
                 [0.983868, 0.904867, 0.136897],
                 [0.993248, 0.906157, 0.143936]]

viridis_r = jnp.array([x[0] for x in _viridis_data])
viridis_g = jnp.array([x[1] for x in _viridis_data])
viridis_b = jnp.array([x[2] for x in _viridis_data])

def viridis(image):
    idxs = jnp.rint(image * 256).astype(jnp.uint32)
    return jnp.stack([viridis_r[idxs], viridis_g[idxs], viridis_b[idxs]], axis=-1)


_MHZ = re.compile(r"(\d+)MHz")


def log_bright_source_flux_comparison(
    dewarped,
    catalog="VLSSR",
    n_sources: int = 10,
    imwcs=None,
    catalog_path=None,
    min_separation_px: int = 15,
    pointlike_axis_ratio_max: float = 1.8,
    bmaj_deg: float | None = None,
    bmin_deg: float | None = None,
):
    """
    Log source position agreement between dewarped-image peaks and a catalog.

    Parameters
    ----------
    dewarped : array-like
        Dewarped image in pixel coordinates.
    catalog : str or array-like
        Either a catalog key (e.g. ``"VLSSR"``/``"NVSS"``) or Nx2 array of
        catalog source positions in pixel coordinates as (x, y).
    n_sources : int
        Target number of catalog sources to compare.
    imwcs : astropy.wcs.WCS, optional
        WCS needed when ``catalog`` is a catalog key string.
    catalog_path : str, optional
        Optional on-disk catalog path forwarded to ``reference_sources``.
    min_separation_px : int
        Minimum pixel separation between selected dewarped peaks.
    pointlike_axis_ratio_max : float
        Maximum major/minor-axis ratio to consider a source point-like when
        morphology columns are available in the source catalog.
    bmaj_deg, bmin_deg : float, optional
        Beam FWHM major/minor axes in degrees. Used to fix Gaussian widths when
        fitting source positions in the dewarped image.
    """
    dewarped_np = np.asarray(dewarped)
    if dewarped_np.ndim != 2:
        raise ValueError("dewarped must be a 2D image.")
    if n_sources <= 0:
        raise ValueError("n_sources must be positive.")

    if isinstance(catalog, str):
        if imwcs is None:
            raise ValueError("`imwcs` is required when `catalog` is a catalog key string.")
        from astropy.wcs.utils import skycoord_to_pixel
        from .catalogs import reference_sources

        source_positions, source_fluxes = reference_sources(
            catalog, min_flux=0, path=catalog_path
        )
        source_ra = np.asarray(source_positions.ra.deg, dtype=float)
        source_dec = np.asarray(source_positions.dec.deg, dtype=float)
        source_gal_b = np.asarray(source_positions.galactic.b.deg, dtype=float)
        catalog_xy = np.stack(skycoord_to_pixel(source_positions, imwcs), axis=1)
        catalog_fluxes = np.asarray(source_fluxes, dtype=float)
        if imwcs.pixel_shape is None:
            scale = 1.0
        else:
            scale = dewarped_np.shape[0] / imwcs.pixel_shape[0]
        catalog_xy = catalog_xy * scale
    else:
        catalog_xy = np.asarray(catalog)
        # If explicit pixel positions are passed, we do not have flux metadata.
        catalog_fluxes = np.full(catalog_xy.shape[0], np.nan, dtype=float)
        source_ra = np.full(catalog_xy.shape[0], np.nan, dtype=float)
        source_dec = np.full(catalog_xy.shape[0], np.nan, dtype=float)
        source_gal_b = np.full(catalog_xy.shape[0], np.nan, dtype=float)

    if catalog_xy.ndim != 2 or catalog_xy.shape[1] != 2:
        raise ValueError("catalog must be a catalog key string or an Nx2 position array.")

    h, w = dewarped_np.shape
    cx = catalog_xy[:, 0]
    cy = catalog_xy[:, 1]
    keep_catalog = (
        np.isfinite(cx)
        & np.isfinite(cy)
        & (cx >= 0)
        & (cx < w)
        & (cy >= 0)
        & (cy < h)
    )
    catalog_xy = catalog_xy[keep_catalog]
    catalog_fluxes = catalog_fluxes[keep_catalog]
    source_ra = source_ra[keep_catalog]
    source_dec = source_dec[keep_catalog]
    source_gal_b = source_gal_b[keep_catalog]
    if catalog_xy.size == 0:
        print("Skipping bright-source position QA: no visible catalog sources.")
        return

    # Build quality mask: point-like morphology (if metadata exists).
    quality_mask = np.ones(catalog_xy.shape[0], dtype=bool)

    if isinstance(catalog, str):
        try:
            import pandas as pd
            from .catalogs import NVSS_CATALOG, VLSSR_CATALOG

            resolved_path = catalog_path
            if resolved_path is None:
                resolved_path = NVSS_CATALOG if catalog == "NVSS" else VLSSR_CATALOG
            if catalog == "NVSS":
                raw = pd.read_csv(resolved_path, sep=r"\s+")
                raw = raw.sort_values(by=["f"])
            else:
                raw = pd.read_csv(resolved_path, sep=" ")
                raw = raw.sort_values(by="PEAK INT")

            cols_upper = {c.upper().replace(" ", "_"): c for c in raw.columns}
            maj_col = None
            min_col = None
            for candidate in ("MAJ", "MAJOR", "BMAJ", "MAJ_AX", "MAJAX"):
                if candidate in cols_upper:
                    maj_col = cols_upper[candidate]
                    break
            for candidate in ("MIN", "MINOR", "BMIN", "MIN_AX", "MINAX"):
                if candidate in cols_upper:
                    min_col = cols_upper[candidate]
                    break

            if (
                maj_col is not None
                and min_col is not None
                and raw.shape[0] >= quality_mask.shape[0]
            ):
                maj = raw[maj_col].to_numpy(dtype=float)
                min_ax = raw[min_col].to_numpy(dtype=float)
                maj = maj[-quality_mask.shape[0] :]
                min_ax = min_ax[-quality_mask.shape[0] :]
                finite_axes = np.isfinite(maj) & np.isfinite(min_ax) & (maj > 0) & (min_ax > 0)
                axis_ratio = np.full(maj.shape, np.inf, dtype=float)
                axis_ratio[finite_axes] = maj[finite_axes] / min_ax[finite_axes]
                quality_mask &= finite_axes & (axis_ratio <= pointlike_axis_ratio_max)
        except Exception:
            # Continue without point-like morphology filtering if metadata parsing fails.
            pass

    catalog_xy = catalog_xy[quality_mask]
    catalog_fluxes = catalog_fluxes[quality_mask]
    source_ra = source_ra[quality_mask]
    source_dec = source_dec[quality_mask]
    source_gal_b = source_gal_b[quality_mask]
    if catalog_xy.size == 0:
        print("Skipping bright-source position QA: no quality-filtered catalog sources.")
        return

    # Select brightest non-extended catalog sources deterministically.
    if np.any(np.isfinite(catalog_fluxes)):
        order = np.lexsort(
            (
                catalog_xy[:, 1],
                catalog_xy[:, 0],
                -np.nan_to_num(catalog_fluxes, nan=-np.inf),
            )
        )
    else:
        order = np.lexsort((catalog_xy[:, 1], catalog_xy[:, 0]))
    take = min(n_sources, catalog_xy.shape[0])
    selected_catalog_xy = catalog_xy[order][:take]
    selected_catalog_fluxes = catalog_fluxes[order][:take]
    selected_catalog_ra = source_ra[order][:take]
    selected_catalog_dec = source_dec[order][:take]

    # Select bright dewarped peaks with simple non-maximum suppression.
    flat_idx = np.argsort(dewarped_np.ravel())[::-1]
    selected_peaks = []
    for idx in flat_idx:
        y, x = np.unravel_index(idx, dewarped_np.shape)
        if not np.isfinite(dewarped_np[y, x]):
            continue
        if selected_peaks:
            peak_xy = np.asarray(selected_peaks, dtype=float)[:, :2]
            deltas = peak_xy - np.array([x, y], dtype=float)
            if np.any(np.linalg.norm(deltas, axis=1) < float(min_separation_px)):
                continue
        selected_peaks.append((float(x), float(y), float(dewarped_np[y, x])))
        if len(selected_peaks) >= max(n_sources, selected_catalog_xy.shape[0]):
            break

    if not selected_peaks:
        print("Skipping bright-source position QA: no finite peaks found in dewarped image.")
        return

    if imwcs is None or bmaj_deg is None or bmin_deg is None:
        raise ValueError(
            "imwcs, bmaj_deg, and bmin_deg are required for Gaussian-fit source matching."
        )
    from astropy.wcs.utils import proj_plane_pixel_scales
    from scipy.optimize import least_squares

    pix_scales = np.abs(proj_plane_pixel_scales(imwcs))
    sigma_y = (float(bmaj_deg) / pix_scales[1]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x = (float(bmin_deg) / pix_scales[0]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    max_match_offset_px = 20.0

    def _fit_centroid_fixed_beam(x0: float, y0: float, amp: float):
        half = int(max(4, np.ceil(3.0 * max(sigma_x, sigma_y))))
        ix = int(np.rint(x0))
        iy = int(np.rint(y0))
        y0w = max(0, iy - half)
        y1w = min(dewarped_np.shape[0], iy + half + 1)
        x0w = max(0, ix - half)
        x1w = min(dewarped_np.shape[1], ix + half + 1)
        patch = dewarped_np[y0w:y1w, x0w:x1w]
        if patch.size == 0 or not np.all(np.isfinite(patch)):
            return np.array([x0, y0], dtype=float)

        yy, xx = np.indices(patch.shape, dtype=float)
        xx += x0w
        yy += y0w

        def residuals(params):
            xc, yc = params
            model = amp * np.exp(
                -0.5 * (((xx - xc) / sigma_x) ** 2 + ((yy - yc) / sigma_y) ** 2)
            )
            return (patch - model).ravel()

        fit = least_squares(
            residuals,
            x0=np.array([x0, y0], dtype=float),
            bounds=(
                np.array([x0w, y0w], dtype=float),
                np.array([x1w - 1, y1w - 1], dtype=float),
            ),
        )
        return fit.x

    print(
        f"Bright-source position QA (catalog={selected_catalog_xy.shape[0]}, "
        f"dewarped_peaks={len(selected_peaks)}):"
    )
    selected_catalog_xy_f = selected_catalog_xy.astype(float)
    selected_peaks_arr = np.asarray(selected_peaks, dtype=float)
    fitted_peak_xy = np.vstack(
        [
            _fit_centroid_fixed_beam(x, y, amp)
            for x, y, amp in selected_peaks_arr
        ]
    )
    print("  idx      ra_deg    dec_deg   cat_flux   sep_pix")
    for i, (x_cat, y_cat) in enumerate(selected_catalog_xy_f):
        deltas = fitted_peak_xy - np.array([x_cat, y_cat], dtype=float)
        dists = np.linalg.norm(deltas, axis=1)
        cat_flux = selected_catalog_fluxes[i]
        ra_deg = selected_catalog_ra[i]
        dec_deg = selected_catalog_dec[i]
        sep = float(np.min(dists))
        if sep > max_match_offset_px:
            sep_out = "no match"
            print(f"  {i:>3d}  {ra_deg:10.5f} {dec_deg:9.5f} {cat_flux:9.3f} {sep_out:>8}")
        else:
            print(f"  {i:>3d}  {ra_deg:10.5f} {dec_deg:9.5f} {cat_flux:9.3f} {sep:8.2f}")


def runqa(
    image,
    reference_sky,
    flow,
    dewarped,
    bright_source_flux_qa_fn=None,
    bright_source_flux_qa_kwargs: Optional[dict] = None,
):
    """
    Run QA checks on a dewarped image and return pass/fail as 1/0.
    """
    offsets = np.nan_to_num(flow.offsets)
    if not offsets.any():
        print("Warning: All offsets zero")

    shift_mag = np.linalg.norm(offsets, axis=2)
    shift_mean = np.mean(shift_mag)
    shift_5, shift_median, shift_95 = np.percentile(shift_mag, [5, 50, 95])
    print(
        f"Shift magnitude mean {shift_mean:.1f} pix "
        f"(5, 50, 95 percentiles: {shift_5:.1f}, {shift_median:.1f}, {shift_95:.1f} pix)"
    )

    score = 1
    if reference_sky is not None:
        pcts = [5, 32, 50, 68, 95]
        residuals = np.abs(np.percentile(dewarped - reference_sky, pcts)) - np.abs(
            np.percentile(image - reference_sky, pcts)
        )
        if not all(residuals < 0):
            print(
                "Not all residuals improved. "
                f"(Percentile, residual difference): {list(zip(pcts, residuals.tolist()))}"
            )
            score = 0

    if bright_source_flux_qa_fn is not None:
        kwargs = {} if bright_source_flux_qa_kwargs is None else bright_source_flux_qa_kwargs
        bright_source_flux_qa_fn(dewarped, **kwargs)
    return score


def group_files_by_frequency(paths: Iterable[str]) -> Dict[int, List[str]]:
    """
    Group file paths by tuning frequency (integer MHz) parsed from each basename.
    """
    out: Dict[int, List[str]] = {}
    for path in paths:
        base = os.path.basename(path)
        m = _MHZ.search(base)
        if m is None:
            raise ValueError(f"no MHz token in basename: {path!r}")
        freq = int(m.group(1))
        out.setdefault(freq, []).append(path)
    return out