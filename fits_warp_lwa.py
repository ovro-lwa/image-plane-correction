from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import wcs

from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bdsf

from multiprocess import Pool

import argparse
import logging
import pickle
import os
import sys
from time import time

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(module)s:%(levelname)s:%(lineno)d %(message)s")
logger.setLevel(logging.INFO)

WORKING_DIR = "working"
OUTPUT_DIR = "outputs"
SAVEFILE_NAME = "parameters.sav"
IMAGE_SIZE = 4096 # assume 4096x4096 images (specific to LWA)

# use half of the CPU cores available
CPU_COUNT = max(1, os.cpu_count() // 2)

# use pybdsf to identify sources from the input image
# using a parameter save file to work around https://github.com/lofar-astron/PyBDSF/issues/232
def identify_sources_bdsf(img, imwcs, min_flux=2.7):
    logger.info(f"Identifying sources in {img} using pybdsf")
    params = {
        "filename": img,
        "outdir": WORKING_DIR,
        "ncores": CPU_COUNT,
    }
    savefile = f"{WORKING_DIR}/{SAVEFILE_NAME}"
    with open(savefile, 'wb') as f:
        pickle.dump(params, f)
        
    result = bdsf.process_image(savefile, quiet=True)
    # re-enable logging since pybdsf removes all existing logger handlers
    logging.basicConfig(format="%(module)s:%(levelname)s:%(lineno)d %(message)s", force=True)

    # take sources above a certain flux density
    positions = np.array([src.posn_sky_centroid for src in result.sources if src.peak_flux_max > min_flux])
    
    # filter out positions that are NaN
    positions = positions[~np.isnan(positions).any(axis=1)]

    # convert to astropy SkyCoord
    sources = SkyCoord(positions, unit=(u.degree, u.degree))
    
    # filter out sources that are near the horizon (not within 70 deg of the center of the image)
    middle = imwcs.pixel_to_world(IMAGE_SIZE//2 - 1, IMAGE_SIZE//2 - 1)  # zero-indexed
    separations = sources.separation(middle)
    filter_idxs = separations < 70 * u.degree
    sources = sources[filter_idxs]

    return sources

# Returns the coordinates of sources in the reference catalog with at least the minimum flux (in mJy)
# The default value is 270 mJy since the NVSS catalog was observed at 1.4 GHz, the LWA testing images
# were taken at ~60 MHz with a lower-bound of ~2.7 Jy, and we assume a spectral index of -0.7.
def reference_sources_nvss(min_flux=270):
    nvss = pd.read_csv("nvss_trim.dat", sep='\s+')
    sorted_nvss = nvss.sort_values(by=['f'])
    
    # cut off refernce sources below a certain flux density
    sorted_nvss = sorted_nvss[sorted_nvss['f'] >= min_flux]

    # get coordinates of each reference source
    nvss_orig = sorted_nvss[["rah", "ram", "ras", "dd", "dm", "ds"]].iloc[:].to_numpy()
    
    # manually convert HMS:DMS into degrees
    nvss_ra = 15 * nvss_orig[:, 0] + (15 / 60) * nvss_orig[:, 1] + (15 / 3600) * nvss_orig[:, 2]
    nvss_dec = nvss_orig[:, 3] + (1/60) * nvss_orig[:, 4] + (1/3600) * nvss_orig[:, 5]
    
    positions = np.stack((nvss_ra, nvss_dec), axis=-1)
    
    return SkyCoord(positions, unit=(u.degree, u.degree))

def crossmatch(sources, ref_sources):
    idx, d2d, d3d = sources.match_to_catalog_sky(ref_sources)
    return ref_sources[idx]

def compute_offsets(dxmodel, dymodel):
    # compute each row separately
    def calc_row(r):
        # all indices with row r
        xy =  np.indices((1, IMAGE_SIZE)).squeeze().transpose()  
        xy[:, 0] = r
        row_offsets = np.stack((dxmodel(xy), dymodel(xy)), axis=-1)
        return row_offsets
    
    # Naive multiprocessing (computing each row separately):
    # Note: while this should be extremely parallelizable , something (likely the GIL)
    # is preventing us from achieving optimal performance. This seems to take about 3
    # minutes with multiprocessing (64 cores) and 4.5 minutes without. Thus, Amdahl's
    # law tells us that only about 25% of this task is parallelizable (though it
    # should be closer to 100%).
    def go():
        res = None
        with Pool(processes=CPU_COUNT) as p:
            try:
                res = p.map(calc_row, list(range(IMAGE_SIZE)))
            except:
                p.close()
                import traceback
                raise Exception("".join(traceback.format_exception(*sys.exc_info())))
        return res

    results = go()
    return np.concatenate(results)

def compute_interpolation(interp):
    def g(r):
        xy =  np.indices((1, IMAGE_SIZE)).squeeze().transpose()
        xy[:, 0] = r
        return interp(xy)
    
    # naive multiprocessing, see above
    def go():
        res = None
        with Pool(processes=CPU_COUNT) as p:
            try:
                res = p.map(g, list(range(IMAGE_SIZE)))
            except:
                p.close()
                import traceback
                raise Exception("".join(traceback.format_exception(*sys.exc_info())))
                
        return res

    results = go()
    interp_img = np.stack(results, axis=0)
    return interp_img

def plot_separations(seps_before, seps_after, output_file=None):
    plt.figure()
    plt.hist([s.arcmin for s in seps_after], bins=100, log=True, fc=(1, 0, 0, 0.7))
    plt.hist([s.arcmin for s in seps_before], bins=100, log=True, fc=(0, 0, 1, 0.7))
    plt.title("Separations before (blue) and after (red) applying dewarping")
    plt.xlabel("Separation (arcmin)")
    plt.ylabel("Frequency")
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()

def plot_image(image_data, title="", output_file=None):
    plt.figure()
    plt.imshow(image_data, interpolation='nearest', origin='lower', vmin=-1, vmax=15)
    plt.title(title)
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()
    

def image_plane_correction(img,
                           smoothing=350,
                           neighbors=20,
                           plot=False,
                          ):
    # get data from fits image
    image = fits.open(img)
    image_data = image[0].data[0, 0, :, :]
    imwcs = wcs.WCS(image[0].header, naxis=2)

    # identify sources from the image using pybdsf
    start = time()
    sources = identify_sources_bdsf(img, imwcs)
    logger.info(f"Done identifying sources in {time() - start} seconds")
    logger.info(f"Found {len(sources)} sources")

    # we are using the NVSS catalog for reference sources
    ref_sources = reference_sources_nvss(min_flux=100)
    logger.info(f"Using {len(ref_sources)} reference sources")

    # cross-match the sources found in the image with the reference sources
    logger.info(f"Crossmatching sources and reference sources")
    matched_ref_sources = crossmatch(sources, ref_sources)
    seps_before = sources.separation(matched_ref_sources)
    logger.info(f"Before correction: median separation of {np.median(seps_before).arcmin} arcmin")

    # pixel coordinates of sources and their corresponding reference sources
    sources_xy = np.stack(wcs.utils.skycoord_to_pixel(sources, imwcs), axis=1)
    ref_xy = np.stack(wcs.utils.skycoord_to_pixel(matched_ref_sources, imwcs), axis=1)

    # offsets between each source and its matched reference
    diff = ref_xy - sources_xy

    # learn an RBF model on the X and Y offsets independently
    # TODO: experiment with different parameters
    logger.info(f"Computing RBF interpolation models")
    dxmodel = RBFInterpolator(sources_xy, diff[:, 0], kernel='linear', smoothing=smoothing, neighbors=neighbors)
    dymodel = RBFInterpolator(sources_xy, diff[:, 1], kernel='linear', smoothing=smoothing, neighbors=neighbors)

    # the interpolated x and y offsets for each pixel, in row-major order
    logger.info(f"Computing offsets at every pixel")
    start = time()
    offsets = compute_offsets(dxmodel, dymodel)  # IMAGE_SIZE^2 x 2
    logger.info(f"Done computing offsets in {time() - start} seconds")

    # add the offset to each image index in the original image to move the pixel to a new location
    logger.info(f"Computing interpolation model for warped pixels")
    start = time()
    image_indices = np.indices((4096, 4096)).swapaxes(0, 2)[:, :, ::-1].reshape((4096 * 4096, 2))
    interp = CloughTocher2DInterpolator(image_indices - offsets, np.ravel(image_data))
    logger.info(f"Done computing interpolation model in {time() - start} seconds")

    # compute interpolated image after applying offsets to each pixel
    logger.info(f"Dewarping the original image")
    start = time()
    dewarped = compute_interpolation(interp)
    logger.info(f"Done dewarping in {time() - start} seconds")

    # write dewarped image to a fits file
    output_img = np.expand_dims(np.expand_dims(dewarped, 0), 0)
    fits.writeto(f"{WORKING_DIR}/temp.fits", output_img, header=image[0].header, overwrite=True)

    # re-compute sources in interpolated image
    start = time()
    new_sources = identify_sources_bdsf(f"{WORKING_DIR}/temp.fits", imwcs)
    logger.info(f"Done identifying new sources in {time() - start} seconds")

    # compute source/reference separations in dewarped image
    new_matches = crossmatch(new_sources, ref_sources)
    seps_after = new_sources.separation(new_matches)
    logger.info(f"After correction: median separation of {np.median(seps_after).arcmin} arcmin")

    if plot:
        plot_separations(seps_before, seps_after, output_file=f"{OUTPUT_DIR}/separations.png")
        plot_image(image_data, "Original image", output_file=f"{OUTPUT_DIR}/original.png")
        plot_image(dewarped, "Dewarped", output_file=f"{OUTPUT_DIR}/dewarped.png")

    # cleanup
    image.close()
    del dxmodel
    del dymodel
    del interp

    # the "score", higher is better
    return np.median(seps_before).arcmin - np.median(seps_after).arcmin

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    group1 = parser.add_argument_group("Warping input/output files")
    group1.add_argument(
        "--img",
        dest="img",
        type=str,
        default=None,
        help="The LWA fits image to be corrected",
    )
    
    group2 = parser.add_argument_group("Plotting")
    
    results = parser.parse_args()
    
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()
        
    if results.img is not None:
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        score = image_plane_correction(results.img, plot=False)
        print(score)
        