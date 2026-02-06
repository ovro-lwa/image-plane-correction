"""Functions for pulling OVRO-LWA input data"""

from typing import Literal, Tuple, Union

import jax.numpy as jnp
from astropy import wcs
from astropy.io import fits
from jaxtyping import Array

from glob import glob


def fits_image(path: str) -> Tuple[Array, wcs.WCS]:
    """
    Reads a fits image at a given path, returning the contained image and wcs data.
    """
    image = fits.open(path)
    image_data = jnp.array(image[0].data.squeeze())
    imwcs = wcs.WCS(image[0].header, naxis=2)
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