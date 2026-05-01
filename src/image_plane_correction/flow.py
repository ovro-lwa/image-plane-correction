from typing import Iterable, Literal, Union, overload

import jax.numpy as jnp
import numpy as np
import os
import logging
from interpax import interp2d
from jaxtyping import Array

from .util import group_files_by_frequency, hsv_to_rgb, indices
from .brox import brox_optical_flow

Direction = Union[Literal["forwards"], Literal["backwards"]]
PHASE2_SUBBANDS = [
    "18MHz",
    "23MHz",
    "27MHz",
    "32MHz",
    "36MHz",
    "41MHz",
    "46MHz",
    "50MHz",
    "55MHz",
    "59MHz",
    "64MHz",
    "69MHz",
    "73MHz",
    "78MHz",
    "82MHz",
]


# inspired by https://github.com/CSRavasio/oflibnumpy
# For now, the flow direction is always "backwards" for performance reasons
# offsets should be of shape (x, y, 2)
class Flow:
    offsets: Array
    direction: Direction

    def __init__(self, offsets, direction: Direction = "backwards"):
        self.offsets = offsets
        self.direction = direction

    def __neg__(self):
        return Flow(-self.offsets)
        
    def __sub__(self, other):
        if isinstance(other, jnp.ndarray):
            return Flow(self.offsets - other)
        return Flow(self.offsets - other.offsets)
        
    def __add__(self, other):
        if isinstance(other, jnp.ndarray):
            return Flow(self.offsets + other)
        return Flow(self.offsets + other.offsets)

    # warps an input single-channel image using bilinear interpolation
    # this function should work on any vector-valued input (e.g. 3D matrices
    # where the first two axes are x/y coordinates.
    def apply(self, image: Array):
        if self.direction == "backwards":
            if len(image.shape) == 2:
                image = jnp.expand_dims(image, axis=-1)

            H, W, C = image.shape

            # H*W query points, one for each pixel in warped image.
            # Offsets are given as as [dx, dy], while images are indexed [y, x], so we need to reverse the last axis of offsets.
            q = jnp.reshape(indices(H, W) + self.offsets[:, :, ::-1], shape=(H * W, 2))

            results = interp2d(
                q[:, 0],
                q[:, 1],
                jnp.arange(H),
                jnp.arange(W),
                image,
                method="linear",
                extrap=0,
            )

            return jnp.reshape(results, shape=image.shape).squeeze()
        else:
            raise NotImplementedError()

    # the flow that would result from applying the current flow and then other_flow
    def compose(self, other_flow):
        return Flow(other_flow.offsets + other_flow.apply(self.offsets))

    @staticmethod
    def zero(shape):
        return Flow(jnp.zeros((shape[0], shape[1], 2)))

    def to_rgb(self, mask=None, scale=None):
        """
        Can pass in a boolean mask of shape (H, W) in to ignore invalid areas in the image.
        """
        if mask is not None:
            flow = self.offsets * jnp.expand_dims(mask, axis=-1)
        else:
            flow = self.offsets

        angle = jnp.arctan2(flow[:, :, 1], flow[:, :, 0])
        angle = (angle + jnp.pi) / (2 * jnp.pi)
        
        magnitude = jnp.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        if scale is None:
            magnitude = magnitude / magnitude.max()
        else:
            magnitude = magnitude / scale

        hsv = jnp.stack([
            angle,
            magnitude,
            jnp.full(angle.shape, 1),
        ], axis=-1)

        return hsv_to_rgb(hsv)

    @staticmethod
    def brox(img1: Array,
             img2: Array,
             alpha=1.0,
             gamma = 150.0,
             scale_factor = 0.7,
             inner_iterations = 5,
             outer_iterations = 150,
             solver_iterations = 10,
             ):
        u_flow, v_flow = brox_optical_flow(
            img2, img1,
            alpha,
            gamma,
            scale_factor,
            inner_iterations,
            outer_iterations,
            solver_iterations,
        )
        uv_flow = jnp.stack([u_flow, v_flow], axis=-1)
        return Flow(uv_flow)

    @staticmethod
    def brox_opencv(img1: Array,
             img2: Array,
             alpha=0.197,
             gamma = 50.0,
             scale_factor = 0.8,
             inner_iterations = 5,
             outer_iterations = 150,
             solver_iterations = 10,
             ):

        import cv2

        denseflow = cv2.cuda.BroxOpticalFlow_create(
            alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations,
        )

        # opencv is currently not directly compatible with JAX arrays, so we convert them to numpy arrays first
        a = cv2.cuda_GpuMat(np.expand_dims(img1, axis=-1).astype(np.float32))
        b = cv2.cuda_GpuMat(np.expand_dims(img2, axis=-1).astype(np.float32))

        flow = denseflow.calc(b, a, None).download()
        
        return Flow(jnp.array(flow))


@overload
def calcflow(
    image_fn,
    psf_fn=None,
    reference_sky=None,
    reference_sky_fn=None,
    cleaned=False,
    qa=True,
    write=False,
    outroot=None,
    catalog="VLSSR",
    max_flux=20,
    catalog_path="/home/claw/vlssr_radecpeak.txt",
    preprocess_weight=1.5,
    alpha=1.3,
    gamma=150,
    scale_factor=0.7,
    use_best_pb_model: bool = False,
    bright_source_flux_qa=False,
    bright_source_flux_qa_count=10,
    return_qa: Literal[True] = True,
): ...


@overload
def calcflow(
    image_fn,
    psf_fn=None,
    reference_sky=None,
    reference_sky_fn=None,
    cleaned=False,
    qa=True,
    write=False,
    outroot=None,
    catalog="VLSSR",
    max_flux=20,
    catalog_path="/home/claw/vlssr_radecpeak.txt",
    preprocess_weight=1.5,
    alpha=1.3,
    gamma=150,
    scale_factor=0.7,
    use_best_pb_model: bool = False,
    bright_source_flux_qa=False,
    bright_source_flux_qa_count=10,
    return_qa: Literal[False] = False,
): ...


def calcflow(
    image_fn,
    psf_fn=None,
    reference_sky=None,
    reference_sky_fn=None,
    cleaned=False,
    qa=True,
    write=False,
    outroot=None,
    catalog="VLSSR",
    max_flux=20,
    catalog_path="/home/claw/vlssr_radecpeak.txt",
    preprocess_weight=1.5,
    alpha=1.3,
    gamma=150,
    scale_factor=0.7,
    use_best_pb_model: bool = False,
    bright_source_flux_qa=False,
    bright_source_flux_qa_count=10,
    return_qa=False,
):
    """
    Compute optical flow and dewarp an image using a theoretical sky model,
    or a precomputed ``reference_sky`` (array or separate FITS via ``reference_sky_fn``).

    Set ``use_best_pb_model=True`` to use the best primary-beam response model
    provided by ``theoretical_sky_beam_function``.
    """
    from astropy.io import fits
    from astropy.wcs.utils import proj_plane_pixel_scales

    from . import data
    from .catalogs import theoretical_sky_beam_function
    from .preprocessing import preprocess
    from .util import log_bright_source_flux_comparison, runqa

    def _sanitize_finite_jax(arr, label: str):
        finite_mask = jnp.isfinite(arr)
        if bool(jnp.all(finite_mask)):
            return arr
        finite_pixels = np.asarray(arr)[np.isfinite(np.asarray(arr))]
        fill_value = float(np.median(finite_pixels)) if finite_pixels.size else 0.0
        out = jnp.array(
            np.nan_to_num(
                np.asarray(arr),
                nan=fill_value,
                posinf=fill_value,
                neginf=fill_value,
            )
        )
        n_bad = int(arr.size - finite_pixels.size)
        print(
            f"Sanitized {n_bad} non-finite pixels in {label} "
            f"using fill value {fill_value:.6g}"
        )
        return out

    print(f"Processing {os.path.basename(image_fn)}")
    if reference_sky is not None and reference_sky_fn is not None:
        raise ValueError("Provide only one of `reference_sky` or `reference_sky_fn`.")
    assert (
        cleaned
        or psf_fn is not None
        or reference_sky is not None
        or reference_sky_fn is not None
    ), "Must provide cleaned=True, PSF file, `reference_sky`, or `reference_sky_fn`"

    image, imwcs = data.fits_image(image_fn)
    image = _sanitize_finite_jax(image, os.path.basename(image_fn))

    if reference_sky_fn is not None:
        reference_sky, _ = data.fits_image(reference_sky_fn)
        reference_sky = _sanitize_finite_jax(
            reference_sky, os.path.basename(reference_sky_fn)
        )

    def _cleaned_psf_from_header(header_path, shape):
        header = fits.getheader(header_path)
        if "BMAJ" not in header or "BMIN" not in header:
            raise KeyError(
                f"Expected BMAJ/BMIN in FITS header for cleaned mode: {header_path}"
            )

        bmaj_deg = float(header["BMAJ"])
        bmin_deg = float(header["BMIN"])
        if bmaj_deg <= 0 or bmin_deg <= 0:
            raise ValueError(
                f"BMAJ/BMIN must be positive in FITS header for cleaned mode: {header_path}"
            )

        # FITS beam sizes are FWHM in degrees; convert to pixel-space sigma.
        pixel_scales = np.abs(proj_plane_pixel_scales(imwcs))
        if pixel_scales.shape[0] < 2:
            raise ValueError(f"Could not derive 2D pixel scales from WCS for: {image_fn}")
        sigma_y = (bmaj_deg / pixel_scales[1]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_x = (bmin_deg / pixel_scales[0]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        h, w = shape
        yy, xx = np.indices((h, w), dtype=np.float64)
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        gaussian = np.exp(
            -0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2)
        )
        peak = gaussian.max()
        if peak <= 0:
            raise ValueError(
                f"Generated cleaned PSF has non-positive peak for: {header_path}"
            )
        gaussian = gaussian / peak
        return jnp.array(gaussian)

    if cleaned:
        # Cleaned images use beam metadata from FITS headers and do not require
        # deriving/reading a matching PSF image by filename.
        if psf_fn is not None:
            with fits.open(psf_fn, memmap=True) as hdul:
                psf_shape = hdul[0].data.squeeze().shape
            header_path = psf_fn
        else:
            # Build a compact synthetic PSF kernel from beam sizes in the image header.
            header = fits.getheader(image_fn)
            if "BMAJ" not in header or "BMIN" not in header:
                raise KeyError(
                    f"Expected BMAJ/BMIN in image header for cleaned mode: {image_fn}"
                )
            pixel_scales = np.abs(proj_plane_pixel_scales(imwcs))
            sigma_y = (float(header["BMAJ"]) / pixel_scales[1]) / (
                2.0 * np.sqrt(2.0 * np.log(2.0))
            )
            sigma_x = (float(header["BMIN"]) / pixel_scales[0]) / (
                2.0 * np.sqrt(2.0 * np.log(2.0))
            )
            radius = int(np.ceil(4.0 * max(sigma_x, sigma_y)))
            size = max(9, 2 * radius + 1)
            psf_shape = (size, size)
            header_path = image_fn
        psf = _cleaned_psf_from_header(header_path, psf_shape)
    elif psf_fn is not None:
        psf, _ = data.fits_image(psf_fn)

    if reference_sky is None:
        image_shape = np.asarray(image).shape
        reference_sky = theoretical_sky_beam_function(
            imwcs,
            psf,
            catalog=catalog,
            img_size=image_shape[0],
            max_flux=max_flux,
            path=catalog_path,
            use_best_pb_model=use_best_pb_model,
        )
    else:
        reference_sky = _sanitize_finite_jax(reference_sky, "reference_sky")

    if np.asarray(image).shape != np.asarray(reference_sky).shape:
        raise ValueError(
            "image and reference_sky must have the same shape: "
            f"{np.asarray(image).shape} vs {np.asarray(reference_sky).shape}"
        )

    image_processed, sky_processed = preprocess(
        image, reference_sky, weight=preprocess_weight
    )
    flow = Flow.brox(
        image_processed,
        sky_processed,
        alpha=alpha,
        gamma=gamma,
        scale_factor=scale_factor,
    )
    dewarped = flow.apply(image)

    if qa:
        bright_source_kwargs = None
        if bright_source_flux_qa:
            beam_header = fits.getheader(image_fn)
            bright_source_kwargs = {
                "imwcs": imwcs,
                "catalog": catalog,
                "catalog_path": catalog_path,
                "n_sources": bright_source_flux_qa_count,
                "bmaj_deg": float(beam_header["BMAJ"]) if "BMAJ" in beam_header else None,
                "bmin_deg": float(beam_header["BMIN"]) if "BMIN" in beam_header else None,
            }
        score = runqa(
            image,
            reference_sky,
            flow,
            dewarped,
            bright_source_flux_qa_fn=log_bright_source_flux_comparison
            if bright_source_flux_qa
            else None,
            bright_source_flux_qa_kwargs=bright_source_kwargs,
        )
    else:
        score = 1

    qa_passed = bool(score == 1)

    if write:
        if outroot is None:
            raise ValueError("`outroot` must be provided when write=True")
        if (qa and qa_passed) or not qa:
            outname = os.path.join(
                outroot, os.path.basename(image_fn.replace(".fits", "_dewarp.fits"))
            )
            dewarped_arr = np.array(dewarped)
            # Preserve full input metadata and refresh WCS-related cards.
            output_header = fits.getheader(image_fn).copy()
            output_header.update(imwcs.to_header())
            fits.writeto(outname, dewarped_arr, output_header)
        else:
            print(f"image {image_fn} failed qa. Not writing dewarped image.")

    if return_qa:
        return image, reference_sky, flow, dewarped, qa_passed
    return image, reference_sky, flow, dewarped


def flow_cascade73MHz(
    image_filenames: Iterable[str],
    psf_filenames: Iterable[str] | None = None,
    cleaned=False,
    qa=True,
    write=False,
    outroot=None,
    catalog="VLSSR",
    max_flux=20,
    catalog_path="/home/claw/vlssr_radecpeak.txt",
    preprocess_weight=1.5,
    alpha=1.3,
    gamma=150,
    scale_factor=0.7,
    use_best_pb_model: bool = False,
    bright_source_flux_qa=False,
    bright_source_flux_qa_count=10,
):
    """
    Run a 73MHz-seeded frequency cascade over a list of image filenames.

    The cascade follows the notebook logic:
    1) Solve flow at 73MHz using the PSF-derived theoretical sky.
    2) Iterate upward in frequency using previous reference sky.
    3) Iterate downward in frequency using previous reference sky.

    Input expectations
    ------------------
    - `image_filenames` contains image-file paths ending with ``-image.fits``.
    - Each image has a matching PSF path via ``-image.fits -> -psf.fits``.
    - For each 73MHz seed filename, string replacement
      ``"73MHz" -> f"{freq}MHz"`` yields existing peer image files across all
      subbands present in `image_filenames`.

    Returns
    -------
    dict[int, list[np.ndarray]]
        Mapping of frequency MHz to a list of flow offset arrays for each 73MHz seed file.
    """
    image_filenames = list(image_filenames)
    groups = group_files_by_frequency(image_filenames)
    freqs = sorted(groups.keys())
    if 73 not in groups:
        raise AssertionError("No 73MHz files found in input group.")
    if not groups[73]:
        raise AssertionError("73MHz group is empty.")

    lookup = {freq: set(paths) for freq, paths in groups.items()}
    psf_by_image = {}
    if not cleaned:
        if psf_filenames is None:
            raise ValueError(
                "For dirty images, `psf_filenames` must be provided and matched to `image_filenames`."
            )
        psf_filenames = list(psf_filenames)
        if len(psf_filenames) != len(image_filenames):
            raise ValueError(
                "`psf_filenames` must have same length/order as `image_filenames`."
            )
        psf_by_image = dict(zip(image_filenames, psf_filenames))
    offsets = {freq: [] for freq in freqs}
    i73 = freqs.index(73)

    for fn in groups[73]:
        if "73MHz" not in fn:
            raise AssertionError(f"Expected '73MHz' token in path: {fn}")

        fn_psf = None if cleaned else psf_by_image.get(fn)
        if not cleaned and fn_psf is None:
            raise ValueError(f"No matching PSF provided for image: {fn}")

        image, reference_sky73, flow, dewarped = calcflow(
            fn,
            psf_fn=fn_psf,
            cleaned=cleaned,
            qa=qa,
            write=write,
            outroot=outroot,
            catalog=catalog,
            max_flux=max_flux,
            catalog_path=catalog_path,
            preprocess_weight=preprocess_weight,
            alpha=alpha,
            gamma=gamma,
            scale_factor=scale_factor,
            use_best_pb_model=use_best_pb_model,
            bright_source_flux_qa=bright_source_flux_qa,
            bright_source_flux_qa_count=bright_source_flux_qa_count,
        )
        offsets[73].append(np.nan_to_num(flow.offsets))

        reference_sky = reference_sky73
        for freq in freqs[i73 + 1 :]:
            fn_next = fn.replace("73MHz", f"{freq}MHz")
            if fn_next == fn or fn_next not in lookup[freq]:
                raise AssertionError(
                    f"Filename replacement did not map to a valid {freq}MHz peer: {fn_next}"
                )
            image, reference_sky, flow, dewarped = calcflow(
                fn_next,
                reference_sky=reference_sky,
                qa=qa,
                write=write,
                outroot=outroot,
                catalog=catalog,
                max_flux=max_flux,
                catalog_path=catalog_path,
                preprocess_weight=preprocess_weight,
                alpha=alpha,
                gamma=gamma,
                scale_factor=scale_factor,
                use_best_pb_model=use_best_pb_model,
                bright_source_flux_qa=bright_source_flux_qa,
                bright_source_flux_qa_count=bright_source_flux_qa_count,
            )
            offsets[freq].append(np.nan_to_num(flow.offsets))

        reference_sky = reference_sky73
        for freq in reversed(freqs[:i73]):
            fn_next = fn.replace("73MHz", f"{freq}MHz")
            if fn_next == fn or fn_next not in lookup[freq]:
                raise AssertionError(
                    f"Filename replacement did not map to a valid {freq}MHz peer: {fn_next}"
                )
            image, reference_sky, flow, dewarped = calcflow(
                fn_next,
                reference_sky=reference_sky,
                qa=qa,
                write=write,
                outroot=outroot,
                catalog=catalog,
                max_flux=max_flux,
                catalog_path=catalog_path,
                preprocess_weight=preprocess_weight,
                alpha=alpha,
                gamma=gamma,
                scale_factor=scale_factor,
                use_best_pb_model=use_best_pb_model,
                bright_source_flux_qa=bright_source_flux_qa,
                bright_source_flux_qa_count=bright_source_flux_qa_count,
            )
            offsets[freq].append(np.nan_to_num(flow.offsets))

    return offsets


def flow_cascade73MHz_phase2(work_dir: str, logger=None):
    """
    Run phase-2 73MHz-seeded cascade over subband directories.

    Parameters
    ----------
    work_dir : str
        Phase 2 working directory containing all subband directories.
    logger : logging.Logger
        Logger to use for logging. If not provided, a default logger will be used.
    Returns
    -------
    dict
        {subband: bool} indicating success/failure per subband.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    all_subbands = list(PHASE2_SUBBANDS)

    image_filenames = []
    psf_by_image = {}
    for sb in all_subbands:
        subband_dir = os.path.join(work_dir, sb)
        if not os.path.isdir(subband_dir):
            continue
        for fname in sorted(os.listdir(subband_dir)):
            if not fname.endswith(".fits"):
                continue
            fn = os.path.join(subband_dir, fname)
            if "-image.fits" not in fname:
                continue
            image_filenames.append(fn)
            psf_candidate = fn.replace("-image.fits", "-psf.fits")
            if os.path.exists(psf_candidate):
                psf_by_image[fn] = psf_candidate

    if not image_filenames:
        logger.warning("No '*-image.fits' files found under %s", work_dir)
        return {sb: False for sb in all_subbands}

    groups = group_files_by_frequency(image_filenames)
    freqs = sorted(groups.keys())
    if 73 not in groups:
        raise AssertionError("No 73MHz files found in input group.")
    if not groups[73]:
        raise AssertionError("73MHz group is empty.")

    lookup = {freq: set(paths) for freq, paths in groups.items()}
    i73 = freqs.index(73)
    results = {}

    def _subband_from_path(path: str) -> str:
        return os.path.relpath(path, work_dir).split(os.sep)[0]

    def _record_result(path: str, qa_ok: bool):
        sb = _subband_from_path(path)
        if sb in results:
            results[sb] = bool(results[sb] and qa_ok)
        else:
            results[sb] = bool(qa_ok)

    for fn in groups[73]:
        if "73MHz" not in fn:
            raise AssertionError(f"Expected '73MHz' token in path: {fn}")

        fn_psf = psf_by_image.get(fn)
        cleaned = fn_psf is None
        _, reference_sky73, _, _, qa_ok = calcflow(
            fn,
            psf_fn=fn_psf,
            cleaned=cleaned,
            qa=True,
            write=False,
            return_qa=True,
        )
        _record_result(fn, qa_ok)

        reference_sky = reference_sky73
        for freq in freqs[i73 + 1 :]:
            fn_next = fn.replace("73MHz", f"{freq}MHz")
            if fn_next == fn or fn_next not in lookup[freq]:
                logger.warning(
                    "Filename replacement did not map to a valid %sMHz peer: %s",
                    freq,
                    fn_next,
                )
                continue
            _, reference_sky, _, _, qa_ok = calcflow(
                fn_next,
                reference_sky=reference_sky,
                qa=True,
                write=False,
                return_qa=True,
            )
            _record_result(fn_next, qa_ok)

        reference_sky = reference_sky73
        for freq in reversed(freqs[:i73]):
            fn_next = fn.replace("73MHz", f"{freq}MHz")
            if fn_next == fn or fn_next not in lookup[freq]:
                logger.warning(
                    "Filename replacement did not map to a valid %sMHz peer: %s",
                    freq,
                    fn_next,
                )
                continue
            _, reference_sky, _, _, qa_ok = calcflow(
                fn_next,
                reference_sky=reference_sky,
                qa=True,
                write=False,
                return_qa=True,
            )
            _record_result(fn_next, qa_ok)

    n_ok = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    n_missing = len(all_subbands) - len(results)
    successful_subbands = [sb for sb, ok in results.items() if ok]
    logger.info(
        "Phase2 cascade QA summary n_ok=%s n_fail=%s n_missing=%s",
        n_ok,
        n_fail,
        n_missing,
    )
    logger.debug("Phase2 successful_subbands=%s", successful_subbands)

    return {sb: bool(results.get(sb, False)) for sb in all_subbands}
