"""Functions for pulling OVRO-LWA input data"""

from typing import Literal, Tuple, Union

import jax.numpy as jnp
import numpy as np
from astropy import wcs
from astropy.io import fits
from jaxtyping import Array

from glob import glob


def fits_image(path: str, squeeze: bool = True) -> Tuple[Array, wcs.WCS]:
    """
    Reads a fits image at a given path, returning the contained image and wcs data.

    Parameters
    ----------
    path
        Path to a FITS file (primary HDU is used).
    squeeze
        If True (default), drop length-1 axes with ``numpy.squeeze`` before returning
        the array (legacy behavior for mostly-2D radio images). If False, return the
        primary HDU array shape unchanged so true 3D/4D cubes stay N-dimensional; the
        returned WCS is still the 2D celestial sub-WCS for the sky plane.
    """
    image = fits.open(path)
    # JAX expects native-endian arrays; FITS data can be big-endian (e.g. '>f4').
    raw = np.asarray(image[0].data)
    image_np = np.squeeze(raw) if squeeze else raw
    if image_np.dtype.byteorder in (">", "!"):
        image_np = image_np.byteswap().view(image_np.dtype.newbyteorder("="))
    image_data = jnp.array(image_np)
    # Many OVRO-LWA products have extra non-celestial axes (e.g. STOKES/FREQ). Always return
    # a 2D celestial WCS so downstream code (skycoord_to_pixel, pixel_to_skycoord, beam maps)
    # behaves consistently on the sky plane (last two array axes).
    full_wcs = wcs.WCS(image[0].header)
    imwcs = full_wcs.celestial
    image.close()
    return image_data, imwcs


def data_1hour(n, order: Union[Literal["time"], Literal["freq"]] = "time"):
    """
    One hour of cleaned OVRO-LWA data, with both frequency and time information
    """
    assert 0 <= n <= 358

    if order == "time":
        t = (n * 36) % 359
    elif order == "freq":
        t = n
    else:
        raise NotImplementedError()

    image, imwcs = fits_image(
        f"/fastpool/zwhuang/data_1hour/alldata.v2.briggs-0.5.tukey30.359intervals-t{t:04d}-image.fits"
    )
    psf, _ = fits_image(
        f"/fastpool/zwhuang/data_1hour/alldata.v2.briggs-0.5.tukey30.359intervals-t{t:04d}-psf.fits"
    )
    return image, psf, imwcs


def data_freq(n):
    """
    A single OVRO-LWA image, separated into 10 sub-bands
    """
    assert 0 <= n <= 9
    image, imwcs = fits_image(
        f"/fastpool/zwhuang/data_freq/alldata.briggs0.tukey30.10channels.v2-{n:04d}-image.fits"
    )
    psf, _ = fits_image(
        f"/fastpool/zwhuang/data_freq/alldata.briggs0.tukey30.10channels.v2-{n:04d}-psf.fits"
    )
    return image, psf, imwcs

    
def data_nivedita(n, get_header=False):
    assert 0 <= n <= 30
    k = 8 + n // 6
    l = n % 6
    image, imwcs = fits_image(
        f"/fastpool/zwhuang/data_nivedita/{k:02d}/46_niter1000_taper30_briggs0-t{l:04d}-I-image.fits"
    )
    data = fits.open(f"/fastpool/zwhuang/data_nivedita/{k:02d}/46_niter1000_taper30_briggs0-t{l:04d}-I-image.fits")
    psf, _ = fits_image(
        f"/fastpool/zwhuang/data_nivedita/{k:02d}/46_niter1000_taper30_briggs0-t{l:04d}-psf.fits"
    )
    if get_header:
        return image, psf, imwcs, data[0].header
    else:
        return image, psf, imwcs
        
def data_nikita(n, get_header=False):
    assert 0 <= n <= 331
    paths = sorted(glob("/fastpool/zwhuang/data_nikita/2025-01-28/09/*-dirty.fits"))
    image, imwcs = fits_image(paths[n])
    psf, _ = fits_image("/fastpool/zwhuang/data_nikita/2025-01-28/09/20250128_090009_41MHz_averaged-psf.fits")
    return image, psf, imwcs
    
def data_pipeline(n, hour, get_header=False):
    assert hour in [3, 4, 13]
    base_path = f"/fastpool/zwhuang/data_nikita/2025-02-02/{str(hour).zfill(2)}"
    paths = sorted(glob(f"{base_path}/*-dirty.fits"))
    assert 0 <= n < len(paths)
    image, imwcs = fits_image(paths[n])
    psf, _ = fits_image(glob(f"{base_path}/*-psf.fits")[0])
    return image, psf, imwcs