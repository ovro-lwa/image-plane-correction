"""Functions for loading true-sky radio sources from various catalogs"""

from typing import Literal, Tuple, Union

import astropy.units as u
import astropy.wcs as wcs
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
    fluxes = fluxes[~jnp.isnan(positions_xy).any(axis=1)]
    positions_xy = positions_xy[~jnp.isnan(positions_xy).any(axis=1)]

    fluxes = jnp.clip(fluxes, 0, max_flux)

    # quantize pixel positions to indices
    xy = jnp.rint(positions_xy).astype(jnp.int32)

    theoretical = jnp.zeros((img_size, img_size))
    theoretical = theoretical.at[xy[:, 1], xy[:, 0]].set(fluxes)

    if perturb is not None:
        theoretical = perturb.apply(theoretical)

    # taper off PSF using a gaussian
    psf_kernel = gkern(img_size, img_size / 8) * psf

    # using FFT as its much faster than a direct 4096x4096 by 4096x4096 convolution
    convolved = convolve(theoretical, psf_kernel, mode="same", method="fft")

    return convolved


# Returns the coordinates of sources in the reference catalog with at least the minimum flux (in mJy)
# The default value is 270 mJy since the NVSS catalog was observed at 1.4 GHz, the LWA testing images
# were taken at ~60 MHz with a lower-bound of ~2.7 Jy, and we assume a spectral index of -0.7.
def reference_sources_nvss(min_flux=270) -> Tuple[SkyCoord, Array]:
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
def reference_sources_vlssr(min_flux=10) -> Tuple[SkyCoord, Array]:
    if path is None:
        path = VLSSR_CATALOG
    vlssr = pd.read_csv(path, sep=" ")
    sorted_vlssr = vlssr.sort_values(by="PEAK INT")
    sorted_vlssr = sorted_vlssr[sorted_vlssr["PEAK INT"] >= min_flux * 10]

    fluxes = sorted_vlssr[["PEAK INT"]].to_numpy().squeeze() / 10

    positions = sorted_vlssr.to_numpy()[:, 0:2]

    return SkyCoord(positions, unit=(u.deg, u.deg)), jnp.array(fluxes)
