"""Functions for loading true-sky radio sources from various catalogs"""

from typing import Literal, Tuple, Union

import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs.utils import pixel_to_skycoord
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from jax.scipy.signal import convolve
from jaxtyping import Array

from flow import Flow
from util import gkern

NVSS_CATALOG = "/fastpool/zwhuang/catalogs/nvss_trim.dat"
VLSSR_CATALOG = "/fastpool/zwhuang/catalogs/vlssr_radecpeak.txt"

Catalog = Union[Literal["NVSS"], Literal["VLSSR"]]


def reference_sources(catalog: Catalog, min_flux=0, path=None) -> Tuple[SkyCoord, Array]:
    """
    Returns the true-sky positions/fluxes associated with reference sources from the NVSS or VLSSR catalogs.
    """
    if catalog == "NVSS":
        return reference_sources_nvss(min_flux, path=path)
    elif catalog == "VLSSR":
        return reference_sources_vlssr(min_flux, path=path)
    else:
        raise NotImplementedError(f"Unknown catalog: {catalog}")


# Note: min_flux is in terms of mJy
# sources are clipped at the max flux level (Jy)
def theoretical_sky(
    imwcs,
    psf,
    perturb: Union[Flow, None] = None,
    catalog: Catalog = "VLSSR",
    img_size=4096,
    min_flux=0,
    max_flux=35,
    path=None
):
    """
    Constructs a theoretical view of the sky using reference sources from a catalog.
    The reference sources are plotted as point sources before being convolved with a given point-spread function.
    A maximum flux is set in order to prevent extremely bright sources from overwhelming the rest of the image.
    If desired, a flow field can be provided that perturbs the point sources before the convolution with the PSF.
    """
    positions, fluxes = reference_sources(catalog, min_flux=min_flux, path=path)

    positions_xy = jnp.stack(wcs.utils.skycoord_to_pixel(positions, imwcs), axis=1)

    # filter out NaNs, e.g. sources not in the field of view
    ignore_positions = jnp.isnan(positions_xy).any(axis=1)
    fluxes = fluxes[~ignore_positions]
    positions_xy = positions_xy[~ignore_positions]

    # scale pixel values to right location (if input imwcs is for larger image than img_size)
    scale = img_size / imwcs.pixel_shape[0]
    positions_xy = positions_xy * scale

    # clip maximum flux (since PSF * large flux tends to overwhelm the image)
    fluxes = jnp.clip(fluxes, 0, max_flux)

    # compute beam function (flux falling off towards edge of image)
    zenith = pixel_to_skycoord(img_size // 2, img_size // 2, imwcs)
    beam_function = jnp.cos(zenith.separation(positions[~ignore_positions]).rad) ** 2.0

    # compute sub-pixel area for each source, assuming samples to be located
    # at the center of each pixel
    half_rounded = jnp.rint(positions_xy + 0.5) - 0.5
    ab = half_rounded - positions_xy + 0.5

    # we compute the area dedicated to the bottom left, bottom right, top left,
    # and top right pixels associated with a square centered at a fractional coordinate
    theoretical = jnp.zeros((img_size, img_size))
    for xy, area in [
        (jnp.trunc(positions_xy), ab[:, 0] * ab[:, 1]),
        (jnp.trunc(positions_xy) + jnp.array([1, 0]), (1 - ab[:, 0]) * ab[:, 1]),
        (jnp.trunc(positions_xy) + jnp.array([0, 1]), ab[:, 0] * (1 - ab[:, 1])),
        (jnp.trunc(positions_xy) + jnp.array([1, 1]), (1 - ab[:, 0]) * (1 - ab[:, 1])),
    ]:
        # adding up flux as delta functions to each pixel area
        idxs = jnp.rint(xy).astype(jnp.int32)
        theoretical = theoretical.at[idxs[:, 1], idxs[:, 0]].add(fluxes * beam_function * area)
    # idxs = jnp.rint(positions_xy).astype(jnp.int32)
    # theoretical = theoretical.at[idxs[:, 1], idxs[:, 0]].set(fluxes * beam_function)

    if perturb is not None:
        theoretical = perturb.apply(theoretical)

    # taper off PSF using a gaussian
    psf_kernel = gkern(psf.shape[0], psf.shape[0] / 4) * psf

    # Convolve point sources with PSF to generate theoretical sky.
    # using FFT as its much faster than a direct 4096x4096 by 4096x4096 convolution
    convolved = convolve(theoretical, psf_kernel, mode="same", method="fft")

    return convolved


# Returns the coordinates of sources in the reference catalog with at least the minimum flux (in mJy)
# The default value is 270 mJy since the NVSS catalog was observed at 1.4 GHz, the LWA testing images
# were taken at ~60 MHz with a lower-bound of ~2.7 Jy, and we assume a spectral index of -0.7.
def reference_sources_nvss(min_flux=270, path=None) -> Tuple[SkyCoord, Array]:
    if path is None:
        path = NVSS_CATALOG
    nvss = pd.read_csv(path, sep=r"\s+")
    sorted_nvss = nvss.sort_values(by=["f"])

    # cut off refernce sources below a certain flux density
    sorted_nvss = sorted_nvss[sorted_nvss["f"] >= min_flux]

    # get coordinates of each reference source
    nvss_orig = sorted_nvss[["rah", "ram", "ras", "dd", "dm", "ds"]].to_numpy()

    # get flux of each reference source in Jy
    fluxes = sorted_nvss[["f"]].to_numpy().squeeze() / 1000

    # manually convert HMS:DMS into degrees
    nvss_ra = (
        15 * nvss_orig[:, 0]
        + (15 / 60) * nvss_orig[:, 1]
        + (15 / 3600) * nvss_orig[:, 2]
    )
    nvss_dec = (
        nvss_orig[:, 3] + (1 / 60) * nvss_orig[:, 4] + (1 / 3600) * nvss_orig[:, 5]
    )

    positions = np.stack((nvss_ra, nvss_dec), axis=-1)

    return SkyCoord(positions, unit=(u.deg, u.deg)), jnp.array(fluxes)


# min_flux should be in terms of mJy, but for some reason the VLSSR intensity
# seems to be in terms of 0.1 mJys.
def reference_sources_vlssr(min_flux=10, path=None) -> Tuple[SkyCoord, Array]:
    if path is None:
        path = VLSSR_CATALOG
    vlssr = pd.read_csv(path, sep=" ")
    sorted_vlssr = vlssr.sort_values(by="PEAK INT")
    sorted_vlssr = sorted_vlssr[sorted_vlssr["PEAK INT"] >= min_flux * 10]

    fluxes = sorted_vlssr[["PEAK INT"]].to_numpy().squeeze() / 10

    positions = sorted_vlssr.to_numpy()[:, 0:2]

    return SkyCoord(positions, unit=(u.deg, u.deg)), jnp.array(fluxes)
