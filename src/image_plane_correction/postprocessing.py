import jax
import jax.numpy as jnp
from jaxtyping import Array
from functools import partial

from .util import rescale_quantile, gaussian_filter, circular_mask, indices

def _tangent_vectors(x, origin):
    """
    Computes (x - origin)^T, e.g. the tangents to the vectors pointing
    from x to the origin.
    """
    x_o = x - origin
    tangents = jnp.stack([-x_o[:, 1], x_o[:, 0]], axis=-1)
    return tangents

def rotational_translational_component(origin, offsets, mask):
    """
    Compute rotational (about some origin) and translational components of a flow
    within a given area. If a significant rotational or translational component
    exists within the flow, then it is a sign of 
    """
    # we will use every point within some mask as sample points
    y = offsets[mask]

    # compute the coordinates of each pixel within the mask
    x_idx = indices(4096, 4096)[mask]

    # for each index, compute the tangent vector to the origin
    t = _tangent_vectors(x_idx, origin)

    # use the tangent vectors and constant 1s as "features" for linear regression
    A = jnp.column_stack((t, jnp.ones(t.shape[0])))

    # perform linear regression in order to determine:
    # can you model the flow as a constant offset plus some coefficient multiplied
    # by a rotational component (tangent vectors to an origin)?
    results = jnp.linalg.lstsq(A, y)

    # compute the strongest rotational/translational component of flow and return it
    new_flow = jnp.zeros_like(offsets)
    new_flow = new_flow.at[x_idx[:,0], x_idx[:, 1], :].set(A @ results[0])
    return new_flow

@partial(jax.jit, static_argnums=(1,))
def _freq_array(sampling=1.0, N=4096):
    f_freq_1d_y = jnp.fft.fftfreq(N, sampling)
    f_freq_1d_x = jnp.fft.fftfreq(N, sampling)
    f_freq_mesh = jnp.meshgrid(f_freq_1d_x, f_freq_1d_y)
    f_freq = jnp.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq

@partial(jax.jit, static_argnums=(1,))
def integrate_2d_fourier(arr, mask=None, sampling=1.0, N=4096):
    """
    Integrates a 2d vector field by performing divison in the fourier domain.
    Expects inputs of shape:
        arr - (N, N, 2)
        mask (optional) - (N, N)

    Reference: https://stackoverflow.com/q/53498672
    """
    if mask is not None:
        arr = arr * jnp.expand_dims(mask, -1)
        
    arr = jnp.transpose(arr, (2, 0, 1))
    
    freqs = _freq_array(sampling, N)

    k_sq = jnp.where(freqs != 0, freqs**2, 0.0001)
    k = jnp.meshgrid(jnp.fft.fftfreq(N, sampling), jnp.fft.fftfreq(N, sampling))

    v_int_x = jnp.real(jnp.fft.ifft2((jnp.fft.fft2(arr[1]) * k[0]) / (2*jnp.pi * 1j * k_sq)))
    v_int_y = jnp.real(jnp.fft.ifft2((jnp.fft.fft2(arr[0]) * k[0]) / (2*jnp.pi * 1j * k_sq)))

    v_int_fs = v_int_x + v_int_y
    return v_int_fs

def gradient(arr):
    return jnp.permute_dims(jnp.stack(jnp.gradient(arr)[::-1]), (1, 2, 0))