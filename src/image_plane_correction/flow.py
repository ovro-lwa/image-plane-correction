from typing import Any, Iterable, Literal, MutableMapping, Optional, Union

import jax
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
    catalog_qa_metrics: Optional[MutableMapping[str, Any]] = None

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

    def apply_broadcast(self, image: Array) -> Array:
        """
        Apply this 2D displacement field to every trailing (row, col) plane.

        Leading dimensions (e.g. Stokes/frequency) are unchanged; each plane is warped
        with the same offsets (typical for a single ionospheric model across a narrow
        spectral cube).
        """
        img = jnp.asarray(image)
        if img.ndim < 2:
            raise ValueError(f"apply_broadcast expects ndim>=2, got shape {img.shape}")
        if img.ndim == 2:
            return self.apply(img)
        lead = img.shape[:-2]
        h, w = int(img.shape[-2]), int(img.shape[-1])
        flat = jnp.reshape(img, (-1, h, w))
        out_flat = jax.vmap(lambda plane: self.apply(plane))(flat)
        return jnp.reshape(out_flat, (*lead, h, w))

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
             alpha=1.1,
             gamma = 125.0,
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


def _obs_date_isot_and_freq_hz_from_header(header):
    """
    Parse observation time as ISOT and frequency in Hz from a FITS header.

    Time: ``DATE-OBS`` (FITS/ISO), else ``MJD-OBS``. Frequency: ``RESTFRQ`` /
    ``RESTFREQ`` (Hz), else the ``FREQ`` WCS axis ``CRVAL`` with ``CUNIT``.
    """
    from astropy import units as u
    from astropy.time import Time

    obs_date_isot = None
    if "DATE-OBS" in header:
        v = header["DATE-OBS"]
        try:
            t = Time(v, format="fits", scale="utc")
        except ValueError:
            t = Time(v, scale="utc")
        obs_date_isot = t.isot
    elif "MJD-OBS" in header:
        obs_date_isot = Time(header["MJD-OBS"], format="mjd", scale="utc").isot

    freq_hz = None
    for key in ("RESTFRQ", "RESTFREQ"):
        if key in header:
            freq_hz = float(header[key])
            break
    if freq_hz is None:
        naxis = int(header.get("NAXIS", 0))
        for i in range(1, naxis + 1):
            ctype = header.get(f"CTYPE{i}", "") or ""
            if not str(ctype).upper().startswith("FREQ"):
                continue
            crval = float(header[f"CRVAL{i}"])
            cunit_raw = header.get(f"CUNIT{i}", "Hz")
            cunit = str(cunit_raw).strip() if cunit_raw is not None else "Hz"
            if not cunit:
                cunit = "Hz"
            freq_hz = (crval * u.Unit(cunit)).to(u.Hz).value
            break

    return obs_date_isot, freq_hz


def horizon_r_normalized(
    imwcs,
    n: int,
    horizon_elevation_deg: Optional[float],
) -> float:
    """
    Normalized disk radius ``r`` passed to :func:`~image_plane_correction.preprocessing.preprocess`
    as ``horizon_r`` (same convention as ``util.circular_mask``).

    If ``horizon_elevation_deg`` is ``None``, returns ``0.7`` (legacy default).
    """
    from astropy.wcs.utils import pixel_to_skycoord

    if horizon_elevation_deg is None:
        return 0.7

    z_deg = 90.0 - float(horizon_elevation_deg)
    if not np.isfinite(z_deg):
        raise ValueError(f"horizon_elevation_deg must be finite, got {horizon_elevation_deg!r}")
    z_deg = float(np.clip(z_deg, 0.0, 180.0))

    cx = cy = (float(n) - 1.0) / 2.0
    c0 = pixel_to_skycoord(cx, cy, imwcs, origin=0)

    rho = np.linspace(0.0, float(n) / 2.0, 2048, dtype=float)
    xs = cx + rho
    ys = np.full_like(xs, cy)
    cs = pixel_to_skycoord(xs, ys, imwcs, origin=0)
    sep_deg = cs.separation(c0).deg

    finite = np.isfinite(sep_deg)
    if not np.any(finite):
        raise ValueError("Could not derive horizon radius from WCS (no finite separations).")
    rho = rho[finite]
    sep_deg = sep_deg[finite]

    order = np.argsort(sep_deg)
    sep_deg = sep_deg[order]
    rho = rho[order]

    z_deg = float(np.clip(z_deg, float(sep_deg[0]), float(sep_deg[-1])))
    rho_z = float(np.interp(z_deg, sep_deg, rho))
    r = (2.0 * rho_z) / float(n)
    return float(np.clip(r, 0.0, 1.0))


def _synth_cleaned_psf_kernel(
    bmaj_deg: float,
    bmin_deg: float,
    imwcs,
    shape: tuple[int, int],
) -> Array:
    """
    Build a synthetic, normalized cleaned-PSF kernel from beam metadata and a WCS.

    This is the shared core of :func:`cleaned_psf_from_fits` and the inline path used
    when an image has been reprojected onto a target grid: in that case the kernel must
    be sized in pixels of the *target* WCS rather than the on-disk WCS.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales

    if bmaj_deg <= 0 or bmin_deg <= 0:
        raise ValueError(
            f"BMAJ/BMIN must be positive for cleaned-PSF synthesis, got "
            f"bmaj={bmaj_deg}, bmin={bmin_deg}"
        )

    pixel_scales = np.abs(proj_plane_pixel_scales(imwcs))
    if pixel_scales.shape[0] < 2:
        raise ValueError("Could not derive 2D pixel scales from WCS for cleaned-PSF synthesis.")
    sigma_y = (bmaj_deg / pixel_scales[1]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x = (bmin_deg / pixel_scales[0]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    gaussian = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
    peak = gaussian.max()
    if peak <= 0:
        raise ValueError("Generated cleaned PSF has non-positive peak.")
    return jnp.array(gaussian / peak)


def cleaned_psf_from_fits(
    fits_path: str,
    *,
    shape: tuple[int, int] | None = None,
    imwcs_override=None,
) -> Array:
    """
    Build a synthetic, normalized PSF image from FITS beam metadata.

    This is intended for "cleaned" images where you have beam FWHM values in the FITS header
    (``BMAJ``/``BMIN`` in degrees), and want a compact PSF image to feed into
    :func:`~image_plane_correction.catalogs.theoretical_sky_beam_function`.

    If ``shape`` is not provided, it is inferred from the header as ``(NAXIS2, NAXIS1)``.

    Pass ``imwcs_override`` to use a different celestial WCS than the one in the file
    (e.g. after reprojecting the parent image onto a different pixel scale). Beam FWHM
    is still read from the FITS header.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    header = fits.getheader(fits_path)
    if "BMAJ" not in header or "BMIN" not in header:
        raise KeyError(f"Expected BMAJ/BMIN in FITS header for cleaned mode: {fits_path}")

    imwcs = imwcs_override if imwcs_override is not None else WCS(header).celestial

    if shape is None:
        if "NAXIS1" not in header or "NAXIS2" not in header:
            raise KeyError(
                f"Expected NAXIS1/NAXIS2 in FITS header to infer PSF shape: {fits_path}"
            )
        shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))

    return _synth_cleaned_psf_kernel(
        bmaj_deg=float(header["BMAJ"]),
        bmin_deg=float(header["BMIN"]),
        imwcs=imwcs,
        shape=shape,
    )


def _make_target_wcs(seed_wcs, seed_n: int, target_size: int):
    """
    Scale a celestial WCS to a ``target_size × target_size`` grid preserving sky coverage.

    Updates ``CRPIX`` to keep pixel centers consistent (FITS 1-indexed) and rescales
    ``CD`` (if present) or ``CDELT`` so the same field of view maps onto the new grid.
    The returned WCS has ``pixel_shape == (target_size, target_size)`` so downstream code
    (e.g. :func:`~image_plane_correction.catalogs.theoretical_sky_beam_function`) sees a
    consistent shape.
    """
    factor = float(seed_n) / float(target_size)  # >1 when downsampling
    target = seed_wcs.deepcopy()
    target.wcs.crpix = (np.asarray(seed_wcs.wcs.crpix, dtype=float) - 0.5) / factor + 0.5
    if seed_wcs.wcs.has_cd():
        target.wcs.cd = np.asarray(seed_wcs.wcs.cd, dtype=float) * factor
    else:
        target.wcs.cdelt = np.asarray(seed_wcs.wcs.cdelt, dtype=float) * factor
    target.pixel_shape = (target_size, target_size)
    return target


_RESAMPLE_METHODS = ("interp", "adaptive", "exact")


def _reproject_array(
    arr,
    source_wcs,
    target_wcs,
    shape_out: tuple[int, int],
    method: str = "interp",
):
    """Reproject ``arr`` from ``source_wcs`` to ``target_wcs`` at the given output shape."""
    if method not in _RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown resample_method {method!r}; expected one of {_RESAMPLE_METHODS}"
        )
    if method == "interp":
        from reproject import reproject_interp as _fn
    elif method == "adaptive":
        from reproject import reproject_adaptive as _fn
    else:
        from reproject import reproject_exact as _fn
    data, _ = _fn((np.asarray(arr), source_wcs), target_wcs, shape_out=shape_out)
    return np.asarray(data)


def _reference_as_hdu(reference, *, fallback_wcs):
    """
    Normalize a reference-sky input (array or HDU) to a ``fits.PrimaryHDU``.

    Array inputs are assumed to live on ``fallback_wcs`` (typically the image's WCS
    *before* any reprojection has been applied). HDU inputs are returned unchanged.
    """
    from astropy.io import fits

    if isinstance(reference, (fits.PrimaryHDU, fits.ImageHDU)):
        return reference
    data = np.asarray(reference)
    return fits.PrimaryHDU(data=data, header=fallback_wcs.to_header())


def _strip_extra_axis_cards(header, max_axis: int = 2) -> None:
    """
    In-place: remove ``NAXISn`` and per-axis WCS cards for ``n > max_axis``.

    Many radio FITS products carry 4-axis WCS metadata (e.g. STOKES, FREQ on axes
    3 and 4) while the image data we operate on is 2D. The celestial WCS we write
    only covers axes 1-2, so leftover cards for the higher axes trigger
    ``VerifyError`` when astropy writes the output. This helper scrubs them.
    """
    import re

    single_axis_re = re.compile(
        r"^(NAXIS|CTYPE|CRVAL|CRPIX|CDELT|CUNIT|CROTA|CNAME|CRDER|CSYER)(\d+)$"
    )
    matrix_re = re.compile(r"^(CD|PC)(\d+)_(\d+)$")
    pv_ps_re = re.compile(r"^(PV|PS)(\d+)_(\d+)$")

    to_delete: list[str] = []
    for key in header:
        m = single_axis_re.match(key)
        if m and int(m.group(2)) > max_axis:
            to_delete.append(key)
            continue
        m = matrix_re.match(key)
        if m and (int(m.group(2)) > max_axis or int(m.group(3)) > max_axis):
            to_delete.append(key)
            continue
        m = pv_ps_re.match(key)
        if m and int(m.group(2)) > max_axis:
            to_delete.append(key)
            continue
    for key in to_delete:
        try:
            del header[key]
        except KeyError:  # pragma: no cover - already gone
            pass


def _collapse_leading_to_spatial_plane(arr: Array) -> Array:
    """Mean over all non-spatial axes (everything before the last two dimensions)."""
    a = jnp.asarray(arr)
    if a.ndim <= 2:
        return a
    axes = tuple(range(a.ndim - 2))
    return jnp.mean(a, axis=axes)


def _spatial_shape_from_array(arr) -> tuple[int, int]:
    a = np.asarray(arr)
    if a.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got shape {a.shape}")
    return (int(a.shape[-2]), int(a.shape[-1]))


def _broadcast_ref_to_image_shape(ref: Array, image_shape: tuple[int, ...]) -> Array:
    """Broadcast 2D ``ref`` to ``image_shape``, or verify full ND ``ref`` matches."""
    r = jnp.asarray(ref)
    spatial = image_shape[-2:]
    if r.ndim == 2:
        if tuple(int(x) for x in r.shape) != spatial:
            raise ValueError(f"2D reference shape {r.shape} != image spatial {spatial}")
        if len(image_shape) == 2:
            return r
        return jnp.broadcast_to(r, image_shape)
    if tuple(int(x) for x in r.shape) != tuple(int(x) for x in image_shape):
        raise ValueError(f"reference shape {r.shape} != image shape {image_shape}")
    return r


def _reproject_nd_trailing_spatial(
    arr: np.ndarray,
    source_wcs,
    target_wcs,
    spatial_shape: tuple[int, int],
    method: str,
) -> np.ndarray:
    """Reproject each trailing spatial plane with the same 2D WCS mapping."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return _reproject_array(arr, source_wcs, target_wcs, spatial_shape, method=method)
    lead = arr.shape[:-2]
    flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    planes = [
        _reproject_array(flat[i], source_wcs, target_wcs, spatial_shape, method=method)
        for i in range(flat.shape[0])
    ]
    return np.stack(planes, axis=0).reshape(*lead, spatial_shape[0], spatial_shape[1])


def _sync_naxis_cards_and_strip_unused_axes(header, shape: tuple[int, ...]) -> None:
    """
    Set FITS ``NAXIS`` / ``NAXISi`` from a numpy/C array shape (last index = FITS axis 1).

    Remove ``NAXISi`` for ``i > len(shape)`` and strip per-axis WCS cards for axes beyond
    the cube dimensionality so the header matches the written array.
    """
    nd = len(shape)
    header["NAXIS"] = nd
    for i in range(1, nd + 1):
        header[f"NAXIS{i}"] = int(shape[-i])
    for i in range(nd + 1, 17):
        k = f"NAXIS{i}"
        if k in header:
            del header[k]
    _strip_extra_axis_cards(header, max_axis=nd)


def _build_preserved_output_header(input_header, imwcs, data_shape: tuple[int, ...]):
    """Merge input metadata with the active 2D celestial WCS and fix ``NAXIS*`` for ``data_shape``."""
    h = input_header.copy()
    h.update(imwcs.to_header())
    _sync_naxis_cards_and_strip_unused_axes(h, data_shape)
    return h


def calcflow(
    image_fn,
    psf_fn=None,
    reference_sky=None,
    reference_sky_fn=None,
    cleaned=False,
    qa=True,
    write=False,
    write_reference_sky: bool = False,
    outroot=None,
    catalog="VLSSR",
    max_flux=20,
    catalog_path="/home/claw/vlssr_radecpeak_unresolved.txt",
    preprocess_weight=1.5,
    horizon_elevation_deg: Optional[float] = 10.0,
    alpha=1.3,
    gamma=150,
    scale_factor=0.7,
    use_best_pb_model: bool = False,
    bright_source_flux_qa=False,
    bright_source_flux_qa_count=10,
    catalog_qa: bool = False,
    catalog_qa_params: Optional[Any] = None,
    quality_metrics: Optional[MutableMapping[str, Any]] = None,
    target_size: Optional[int] = None,
    target_wcs: Optional[Any] = None,
    resample_method: Literal["interp", "adaptive", "exact"] = "interp",
):
    """
    Compute optical flow and dewarp an image using a theoretical sky model,
    or a precomputed ``reference_sky`` (array or HDU, or separate FITS via ``reference_sky_fn``).

    Brox smoothness ``alpha`` and gradient ``gamma`` default to values used historically in
    this pipeline; dataset-specific tuning can be done with ``scripts/optimize_alpha_gamma.py``
    (coarse log-space ``--search`` and optional ``--refine``).

    Set ``use_best_pb_model=True`` to use the best primary-beam response model
    provided by ``theoretical_sky_beam_function``. Observation time (ISOT) and
    frequency (Hz) are read from the image FITS header—``DATE-OBS`` / ``MJD-OBS``
    and ``RESTFRQ`` / ``RESTFREQ`` or a ``FREQ`` axis—and passed through so the
    beam model matches the observation.

    When ``cleaned=True``, ``reference_sky`` is run through the same non-finite
    sanitization as the image after it is built or supplied (e.g. NaNs from
    beam/WCS outside the physical sky).

    Set ``catalog_qa=True`` to run catalog astrometry QA on raw vs dewarped images using
    PyBDSF for source measurement (see :mod:`image_plane_correction.qa`). Results are merged into ``quality_metrics``
    when that mapping is provided; otherwise they are stored only on ``flow.catalog_qa_metrics``.
    ``catalog_qa_params=None`` uses default thresholds (:class:`~image_plane_correction.qa.CatalogAstrometryQAParams`).

    Set ``write_reference_sky=True`` to write the reference sky map used for flow solving to
    ``outroot`` using a filename suffix ``"_reference.fits"`` (gated by the same QA pass
    condition as ``write=True`` output products).

    Resampling
    ----------
    Pass ``target_size`` (or an explicit ``target_wcs``) to resample the image and
    reference sky onto a common ``N×N`` grid before flow solving. This lets ``calcflow``
    accept a pair of inputs with different sizes — useful for the frequency cascade,
    where subbands often have different pixel scales. When ``target_wcs`` is omitted,
    it is built from the image's WCS scaled to ``target_size``, preserving the same
    field of view. Resampling uses the ``reproject`` library; pick the algorithm via
    ``resample_method`` (``"interp"`` for bilinear / default, ``"adaptive"`` for
    anti-aliased downsampling, ``"exact"`` for flux-conserving spherical-polygon
    overlap).

    Multi-dimensional images
    ------------------------
    The image is read without collapsing singleton axes. Data with shape
    ``(..., n_y, n_x)`` keep their leading dimensions through dewarping: Brox flow is
    solved on the mean over leading planes (one shared 2D warp), then that field is
    applied to every plane. Written FITS headers merge the input metadata with the
    updated celestial WCS and retain extra axes (e.g. ``FREQ``) when the array is a
    true 3D/4D cube.

    The returned ``reference_sky`` is always a :class:`~astropy.io.fits.PrimaryHDU`
    that carries both the array data and the WCS used during flow solving, which
    makes it self-describing for use in chained calls (e.g. the frequency cascades).
    """
    from astropy.io import fits

    from . import data
    from .catalogs import theoretical_sky_beam_function
    from .preprocessing import preprocess
    from .qa import (
        CatalogAstrometryQAParams,
        bright_source_qa_kwargs,
        catalog_astrometry_metrics_pair,
        check_reference_sky,
        log_bright_source_alignment,
    )
    from .util import runqa

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

    input_header = fits.getheader(image_fn)
    obs_date, freq_hz = _obs_date_isot_and_freq_hz_from_header(input_header)
    if use_best_pb_model and (obs_date is None or freq_hz is None):
        raise ValueError(
            "use_best_pb_model requires observation time and frequency in the image header "
            "(e.g. DATE-OBS and RESTFRQ, or a FREQ WCS axis). "
            f"Could not parse for {image_fn!r}: obs_date={obs_date!r}, freq_hz={freq_hz!r}."
        )

    image, source_imwcs = data.fits_image(image_fn, squeeze=False)
    image = _sanitize_finite_jax(image, os.path.basename(image_fn))
    source_image_spatial_shape = _spatial_shape_from_array(image)

    # Resolve the target grid (if any). When neither target_size nor target_wcs is
    # provided, no reprojection happens and everything operates on the image's native grid.
    if target_wcs is not None and target_size is None:
        ps = getattr(target_wcs, "pixel_shape", None)
        if ps is None or ps[0] is None or ps[1] is None:
            raise ValueError(
                "`target_wcs` must have its `pixel_shape` set, or pass `target_size` explicitly."
            )
        if int(ps[0]) != int(ps[1]):
            raise ValueError(
                f"`target_wcs.pixel_shape` must be square for calcflow, got {ps}."
            )
        target_size = int(ps[0])
    if target_size is not None and target_wcs is None:
        target_wcs = _make_target_wcs(
            source_imwcs, source_image_spatial_shape[0], int(target_size)
        )

    if target_size is not None:
        assert target_wcs is not None  # resolved above
        target_shape = (int(target_size), int(target_size))
        if source_image_spatial_shape != target_shape:
            image_np = _reproject_nd_trailing_spatial(
                np.asarray(image),
                source_imwcs,
                target_wcs,
                target_shape,
                method=resample_method,
            )
            image = _sanitize_finite_jax(
                jnp.array(image_np), f"{os.path.basename(image_fn)} (reprojected)"
            )
        imwcs = target_wcs
    else:
        imwcs = source_imwcs

    from astropy.wcs import WCS as _WCS

    image_shape = tuple(int(s) for s in np.asarray(image).shape)
    spatial_shape = _spatial_shape_from_array(image)

    # Normalize reference_sky inputs to an HDU on `imwcs`. We track whether the
    # caller supplied a reference so we can decide whether to build one fresh.
    reference_hdu: Optional[Any] = None
    if reference_sky_fn is not None:
        with fits.open(reference_sky_fn) as hdul:
            primary = hdul[0]
            ref_data = np.asarray(primary.data)
            if ref_data.dtype.byteorder in (">", "!"):
                ref_data = ref_data.byteswap().view(ref_data.dtype.newbyteorder("="))
            ref_hdr_in = primary.header.copy()
            reference_hdu = fits.PrimaryHDU(data=ref_data, header=ref_hdr_in)
    elif reference_sky is not None:
        # Array inputs are assumed to live on the *source* image's WCS (pre-reprojection).
        # HDU inputs carry their own WCS.
        reference_hdu = _reference_as_hdu(reference_sky, fallback_wcs=source_imwcs)

    if reference_hdu is not None:
        ref_wcs = _WCS(reference_hdu.header).celestial
        ref_label = os.path.basename(reference_sky_fn) if reference_sky_fn else "array"
        ref_data = _sanitize_finite_jax(
            jnp.array(np.asarray(reference_hdu.data)),
            f"reference_sky ({ref_label})",
        )
        ref_spatial = _spatial_shape_from_array(ref_data)
        # Match spatial grid to the active image; broadcast 2D references across leading axes.
        if ref_data.ndim == 2:
            if ref_spatial != spatial_shape:
                ref_np = _reproject_array(
                    np.asarray(ref_data),
                    ref_wcs,
                    imwcs,
                    spatial_shape,
                    method=resample_method,
                )
                ref_data = _sanitize_finite_jax(
                    jnp.array(ref_np), "reference_sky (reprojected)"
                )
            ref_data = _broadcast_ref_to_image_shape(ref_data, image_shape)
        else:
            if tuple(int(x) for x in ref_data.shape[:-2]) != tuple(image_shape[:-2]):
                raise ValueError(
                    "reference_sky leading axes must match the image "
                    f"(ref {tuple(ref_data.shape)} vs image {image_shape})"
                )
            if ref_spatial != spatial_shape:
                ref_np = _reproject_nd_trailing_spatial(
                    np.asarray(ref_data),
                    ref_wcs,
                    imwcs,
                    spatial_shape,
                    method=resample_method,
                )
                ref_data = _sanitize_finite_jax(
                    jnp.array(ref_np), "reference_sky (reprojected)"
                )
        ref_header = _build_preserved_output_header(
            input_header, imwcs, tuple(int(s) for s in np.asarray(ref_data).shape)
        )
        reference_hdu = fits.PrimaryHDU(data=np.asarray(ref_data), header=ref_header)

    # Build the PSF kernel on the *active* (post-reproject) grid when needed.
    psf = None
    if cleaned:
        # Cleaned images use beam metadata from FITS headers; the kernel must live
        # at the same pixel scale as the image being analyzed. When ``psf_fn`` is
        # supplied, beam keys are read from that file's header; otherwise from the
        # image header.
        beam_header = fits.getheader(psf_fn) if psf_fn is not None else input_header
        if "BMAJ" not in beam_header or "BMIN" not in beam_header:
            raise KeyError(
                f"Expected BMAJ/BMIN in {'PSF' if psf_fn is not None else 'image'} "
                f"header for cleaned mode: {psf_fn or image_fn}"
            )
        bmaj_deg = float(beam_header["BMAJ"])
        bmin_deg = float(beam_header["BMIN"])
        from astropy.wcs.utils import proj_plane_pixel_scales

        pixel_scales = np.abs(proj_plane_pixel_scales(imwcs))
        sigma_y = (bmaj_deg / pixel_scales[1]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_x = (bmin_deg / pixel_scales[0]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        radius = int(np.ceil(4.0 * max(sigma_x, sigma_y)))
        # Preserve the legacy behavior of inferring the PSF shape from the PSF file's
        # ``NAXIS`` when one is provided and no reprojection has occurred.
        if psf_fn is not None and target_size is None:
            psf = cleaned_psf_from_fits(psf_fn)
        else:
            size = max(9, 2 * radius + 1)
            psf = _synth_cleaned_psf_kernel(
                bmaj_deg=bmaj_deg,
                bmin_deg=bmin_deg,
                imwcs=imwcs,
                shape=(size, size),
            )
    elif psf_fn is not None:
        psf_np, psf_wcs = data.fits_image(psf_fn)
        psf_np = np.asarray(psf_np)
        if target_size is not None and int(target_size) != int(source_image_spatial_shape[0]):
            # Scale the PSF's WCS by the same factor as the image so its kernel covers
            # the same on-sky support at the new pixel scale. The PSF array itself is
            # resized by the same factor so its footprint (in pixels of the analysis grid)
            # is preserved.
            factor = float(source_image_spatial_shape[0]) / float(target_size)  # >1 when downsampling
            new_psf_n = max(9, int(round(psf_np.shape[0] / factor)))
            if new_psf_n % 2 == 0:
                new_psf_n += 1
            psf_target_wcs = _make_target_wcs(psf_wcs, psf_np.shape[0], new_psf_n)
            psf_np = _reproject_array(
                psf_np, psf_wcs, psf_target_wcs, (new_psf_n, new_psf_n), method=resample_method
            )
            psf_np = np.nan_to_num(psf_np, nan=0.0, posinf=0.0, neginf=0.0)
        psf = jnp.array(psf_np)

    if reference_hdu is None:
        # Build a fresh reference sky on the active grid using the catalog + PSF.
        ref_arr = theoretical_sky_beam_function(
            imwcs,
            psf,
            catalog=catalog,
            img_size=spatial_shape[0],
            max_flux=max_flux,
            path=catalog_path,
            use_best_pb_model=use_best_pb_model,
            obs_date=obs_date,
            freq_hz=freq_hz,
        )
        ref_arr = _sanitize_finite_jax(ref_arr, "reference_sky (synthetic)")
        ref_arr = _broadcast_ref_to_image_shape(ref_arr, image_shape)
        ref_header = _build_preserved_output_header(
            input_header, imwcs, tuple(int(s) for s in np.asarray(ref_arr).shape)
        )
        reference_hdu = fits.PrimaryHDU(data=np.asarray(ref_arr), header=ref_header)
    assert reference_hdu is not None  # narrowed for downstream attribute access

    if np.asarray(image).shape != np.asarray(reference_hdu.data).shape:
        raise ValueError(
            "image and reference_sky must have the same shape after any resampling: "
            f"{np.asarray(image).shape} vs {np.asarray(reference_hdu.data).shape}"
        )

    # Validate reference sky before it is used for preprocessing / flow solving.
    check_reference_sky(reference_hdu, label="reference_sky")

    n_img = int(spatial_shape[0])
    horizon_r = horizon_r_normalized(imwcs, n=n_img, horizon_elevation_deg=horizon_elevation_deg)

    reference_jax = jnp.array(reference_hdu.data)
    image_2d = _collapse_leading_to_spatial_plane(image)
    ref_2d = _collapse_leading_to_spatial_plane(reference_jax)
    image_processed, sky_processed = preprocess(
        image_2d, ref_2d, weight=preprocess_weight, horizon_r=horizon_r
    )
    flow = Flow.brox(
        image_processed,
        sky_processed,
        alpha=alpha,
        gamma=gamma,
        scale_factor=scale_factor,
    )
    dewarped = flow.apply_broadcast(image)

    if qa:
        bright_source_kwargs = None
        if bright_source_flux_qa:
            bright_source_kwargs = bright_source_qa_kwargs(
                imwcs,
                input_header,
                catalog=catalog,
                catalog_path=catalog_path,
                n_sources=bright_source_flux_qa_count,
            )
        bright_fn = None
        if bright_source_flux_qa:
            if int(np.asarray(dewarped).ndim) > 2:

                def bright_fn(d, **kwargs):
                    d2 = np.asarray(_collapse_leading_to_spatial_plane(jnp.asarray(d)))
                    return log_bright_source_alignment(d2, **kwargs)

            else:
                bright_fn = log_bright_source_alignment
        score = runqa(
            image,
            reference_jax,
            flow,
            dewarped,
            bright_source_flux_qa_fn=bright_fn,
            bright_source_flux_qa_kwargs=bright_source_kwargs,
        )
    else:
        score = 1

    qa_passed = bool(score == 1)

    if catalog_qa:
        sink: MutableMapping[str, Any] = quality_metrics if quality_metrics is not None else {}
        params = (
            catalog_qa_params
            if catalog_qa_params is not None
            else CatalogAstrometryQAParams()
        )
        qa_row = catalog_astrometry_metrics_pair(
            np.asarray(_collapse_leading_to_spatial_plane(jnp.asarray(image))),
            np.asarray(_collapse_leading_to_spatial_plane(jnp.asarray(dewarped))),
            imwcs,
            catalog,
            catalog_path,
            image_fn,
            params,
        )
        sink.update(qa_row)
        flow.catalog_qa_metrics = sink

    if write or write_reference_sky:
        if outroot is None:
            raise ValueError("`outroot` must be provided when write=True or write_reference_sky=True")
        if (qa and qa_passed) or not qa:
            dewarped_arr = np.asarray(dewarped)
            output_header = _build_preserved_output_header(
                input_header, imwcs, tuple(int(s) for s in dewarped_arr.shape)
            )

            if write:
                outname = os.path.join(
                    outroot, os.path.basename(image_fn.replace(".fits", "_dewarp.fits"))
                )
                fits.writeto(outname, dewarped_arr, output_header)

            if write_reference_sky:
                ref_outname = os.path.join(
                    outroot, os.path.basename(image_fn.replace(".fits", "_reference.fits"))
                )
                fits.writeto(
                    ref_outname, np.asarray(reference_hdu.data), output_header
                )
        else:
            msg = f"image {image_fn} failed qa. Not writing outputs."
            print(msg)

    return image, reference_hdu, flow, dewarped, qa_passed


def flow_cascade73MHz(
    image_filenames: Iterable[str],
    psf_filenames: Optional[Iterable[str]] = None,
    cleaned=False,
    qa=True,
    write=False,
    outroot=None,
    catalog="VLSSR",
    max_flux=20,
    catalog_path="/home/claw/vlssr_radecpeak_unresolved.txt",
    preprocess_weight=1.5,
    alpha=1.3,
    gamma=150,
    scale_factor=0.7,
    use_best_pb_model: bool = False,
    bright_source_flux_qa=False,
    bright_source_flux_qa_count=10,
    target_size: Optional[int] = None,
    resample_method: Literal["interp", "adaptive", "exact"] = "interp",
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

    Resampling
    ----------
    Pass ``target_size`` (or rely on auto-discovery from the 73 MHz seed when ``None``)
    to put every subband onto a common ``N×N`` grid. When ``None``, the seed image's
    ``NAXIS1`` defines the canonical grid — appropriate when the 73 MHz seed is the
    largest input, since all later steps then downsample onto it.

    Returns
    -------
    dict[int, list[np.ndarray]]
        Mapping of frequency MHz to a list of flow offset arrays for each 73MHz seed file.
    """
    from astropy.io import fits

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

        seed_target_size: Optional[int] = target_size
        if seed_target_size is None:
            seed_header = fits.getheader(fn)
            if "NAXIS1" in seed_header:
                seed_target_size = int(seed_header["NAXIS1"])

        image, reference_sky73, flow, dewarped, _qa_ok = calcflow(
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
            target_size=seed_target_size,
            resample_method=resample_method,
        )
        offsets[73].append(np.nan_to_num(flow.offsets))

        for direction_freqs in (freqs[i73 + 1 :], list(reversed(freqs[:i73]))):
            for freq in direction_freqs:
                fn_next = fn.replace("73MHz", f"{freq}MHz")
                if fn_next == fn or fn_next not in lookup[freq]:
                    raise AssertionError(
                        f"Filename replacement did not map to a valid {freq}MHz peer: {fn_next}"
                    )
                image, _, flow, dewarped, _qa_ok = calcflow(
                    fn_next,
                    reference_sky=reference_sky73,
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
                    target_size=seed_target_size,
                    resample_method=resample_method,
                )
                offsets[freq].append(np.nan_to_num(flow.offsets))

    return offsets


def flow_cascade73MHz_phase2(
    work_dir: str,
    logger=None,
    *,
    target_size: Optional[int] = None,
    resample_method: Literal["interp", "adaptive", "exact"] = "interp",
):
    """
    Run phase-2 73MHz-seeded cascade over subband directories.

    Parameters
    ----------
    work_dir : str
        Phase 2 working directory containing all subband directories.
    logger : logging.Logger
        Logger to use for logging. If not provided, a default logger will be used.
    target_size : int, optional
        Force all subbands onto a common ``N×N`` grid via WCS-aware reprojection.
        When ``None`` (default), the size is auto-discovered from the 73 MHz seed
        image's ``NAXIS1`` and the cascade downsamples every other subband onto it.
    resample_method : {"interp", "adaptive", "exact"}, default "interp"
        Resampling algorithm forwarded to :mod:`reproject`. ``"interp"`` is bilinear,
        ``"adaptive"`` is DeForest anti-aliased, ``"exact"`` is flux-conserving.

    Returns
    -------
    dict
        {subband: bool} indicating success/failure per subband.
    """
    from astropy.io import fits

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

        seed_target_size: Optional[int] = target_size
        if seed_target_size is None:
            try:
                seed_header = fits.getheader(fn)
                if "NAXIS1" in seed_header:
                    seed_target_size = int(seed_header["NAXIS1"])
            except Exception as exc:  # pragma: no cover - defensive: keep cascade running
                logger.warning("Could not read seed header to derive target_size: %s", exc)
                seed_target_size = None

        _, reference_sky73, _, _, qa_ok = calcflow(
            fn,
            psf_fn=fn_psf,
            cleaned=cleaned,
            qa=True,
            write=False,
            target_size=seed_target_size,
            resample_method=resample_method,
        )
        _record_result(fn, qa_ok)

        for direction_freqs in (freqs[i73 + 1 :], list(reversed(freqs[:i73]))):
            for freq in direction_freqs:
                fn_next = fn.replace("73MHz", f"{freq}MHz")
                if fn_next == fn or fn_next not in lookup[freq]:
                    logger.warning(
                        "Filename replacement did not map to a valid %sMHz peer: %s",
                        freq,
                        fn_next,
                    )
                    continue
                _, _, _, _, qa_ok = calcflow(
                    fn_next,
                    reference_sky=reference_sky73,
                    qa=True,
                    write=False,
                    target_size=seed_target_size,
                    resample_method=resample_method,
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
