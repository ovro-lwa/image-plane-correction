import logging
import pickle

import astropy.units as u
import bdsf
import numpy as np
from astropy.coordinates import SkyCoord

SAVEFILE_NAME = "parameters.sav"

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(module)s:%(levelname)s:%(lineno)d %(message)s")
logger.setLevel(logging.INFO)


# use pybdsf to identify sources from the input image
# Using a parameter save file to work around https://github.com/lofar-astron/PyBDSF/issues/232
# returns a SkyCoord containing all of the identified source positions
# Note, writes intermediate files to work_dir
def identify_sources_bdsf(img, imwcs, work_dir, N=4096, min_flux=2.7) -> SkyCoord:
    logger.info(f"Identifying sources in {img} using pybdsf")
    params = {
        "filename": img,
        "outdir": work_dir,
        "mean_map": "const",
        "thresh": "hard",
        "thresh_isl": 100,
        "thresh_pix": 5,
    }
    savefile = f"{work_dir}/{SAVEFILE_NAME}"
    with open(savefile, "wb") as f:
        pickle.dump(params, f)

    result = bdsf.process_image(savefile, quiet=True)
    # re-enable logging since pybdsf removes all existing logger handlers
    # https://github.com/lofar-astron/PyBDSF/issues/233
    logging.basicConfig(
        format="%(module)s:%(levelname)s:%(lineno)d %(message)s", force=True
    )

    # take sources above a certain flux density
    positions = np.array(
        [
            src.posn_sky_centroid
            for src in result.sources
            if src.peak_flux_max > min_flux
        ]
    )

    # filter out positions that are NaN
    positions = positions[~np.isnan(positions).any(axis=1)]

    # convert to astropy SkyCoord
    sources = SkyCoord(positions, unit=(u.deg, u.deg))

    # filter out sources that are near the horizon (not within 70 deg of the center of the image)
    middle = imwcs.pixel_to_world(N // 2 - 1, N // 2 - 1)  # zero-indexed
    separations = sources.separation(middle)
    filter_idxs = separations < 70 * u.deg
    sources = sources[filter_idxs]

    return sources
