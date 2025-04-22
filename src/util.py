"""Utility functions (mostly ported from other libraries to JAX)"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array


def indices(m, n):
    return jnp.indices((m, n)).transpose(1, 2, 0)


# https://github.com/google/jax/pull/15359
def _gaussian_kernel1d(sigma: float, order: int, radius: int) -> Array:
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = jnp.arange(order + 1)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = jnp.zeros(order + 1)
        q = q.at[0].set(1)
        D = jnp.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = jnp.diag(jnp.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def gaussian_filter1d(
    input: Array,
    sigma: float,
    axis=-1,
    order=0,
    truncate=4.0,
    *,
    radius: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    precision=None,
) -> Array:
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)

    if mode != "constant" or cval != 0.0:
        raise NotImplementedError(
            'Other modes than "constant" with 0. fill value are not' "supported yet."
        )

    if radius > 0.0:
        lw = radius
    if lw < 0:
        raise ValueError("Radius must be a nonnegative integer.")

    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

    # Be careful that modes in signal.convolve refer to the 'same' 'full' 'valid' modes
    # while in gaussian_filter1d refers to the way the padding is done 'constant' 'reflect' etc.
    # We should change the convolve backend for further features
    return jnp.apply_along_axis(
        jsp.signal.convolve,
        axis,
        input,
        weights,
        mode="same",
        method="auto",
        precision=precision,
    )


def gaussian_filter(
    input: Array,
    sigma: float,
    order: int = 0,
    truncate: float = 4.0,
    *,
    radius: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    precision=None,
) -> Array:
    input = jnp.asarray(input)

    for axis in range(input.ndim):
        input = gaussian_filter1d(
            input,
            sigma,
            axis=axis,
            order=order,
            truncate=truncate,
            radius=radius,
            mode=mode,
            precision=precision,
            cval=cval,
        )

    return input


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
