"""Reuse synthetic ``reference_sky`` HDUs across sequential ``calcflow`` calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


def obs_time_from_header(header: fits.Header) -> Optional[str]:
    """
    Observation epoch as ISOT from ``DATE-OBS`` or ``MJD-OBS`` (same convention as calcflow).

    Returns ``None`` when neither card is present.
    """
    if "DATE-OBS" in header:
        v = header["DATE-OBS"]
        try:
            t = Time(v, format="fits", scale="utc")
        except ValueError:
            t = Time(v, scale="utc")
        return t.isot
    if "MJD-OBS" in header:
        return Time(header["MJD-OBS"], format="mjd", scale="utc").isot
    return None


def celestial_wcs_fingerprint(header: fits.Header) -> Tuple[Hashable, ...]:
    """
    Compact fingerprint of the celestial WCS (pointing + pixel scale + image size).

    Used to invalidate a cached reference when the sky grid changes.
    """
    w = WCS(header).celestial
    crval = tuple(float(x) for x in np.atleast_1d(w.wcs.crval)[:2])
    crpix = tuple(float(x) for x in np.atleast_1d(w.wcs.crpix)[:2])
    cdelt = tuple(float(x) for x in np.atleast_1d(w.wcs.cdelt)[:2])
    ctype = tuple(str(x) for x in np.atleast_1d(w.wcs.ctype)[:2])
    naxis = (
        int(header.get("NAXIS1", 0) or 0),
        int(header.get("NAXIS2", 0) or 0),
    )
    return crval, crpix, cdelt, ctype, naxis


@dataclass
class ReferenceSkyCache:
    """
    In-process cache of the last synthetic reference-sky HDU.

    Cache key: celestial WCS fingerprint, ``target_size``, and observation time
    (``DATE-OBS`` / ``MJD-OBS``). Hit → pass ``reference_sky`` into ``calcflow``
    and skip ``theoretical_sky_beam_function``.
    """

    _key: Optional[Tuple[Hashable, ...]] = field(default=None, init=False, repr=False)
    _hdu: Optional[fits.PrimaryHDU] = field(default=None, init=False, repr=False)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def make_key(
        *,
        header: fits.Header,
        target_size: Optional[int],
    ) -> Tuple[Hashable, ...]:
        return (
            celestial_wcs_fingerprint(header),
            None if target_size is None else int(target_size),
            obs_time_from_header(header),
        )

    def get(self, key: Tuple[Hashable, ...]) -> Optional[fits.PrimaryHDU]:
        if self._key == key and self._hdu is not None:
            self.hits += 1
            return self._hdu
        self.misses += 1
        return None

    def store(self, key: Tuple[Hashable, ...], hdu: Any) -> fits.PrimaryHDU:
        data = np.array(np.asarray(hdu.data), copy=True)
        header = hdu.header.copy() if getattr(hdu, "header", None) is not None else fits.Header()
        stored = fits.PrimaryHDU(data=data, header=header)
        self._key = key
        self._hdu = stored
        return stored

    def clear(self) -> None:
        self._key = None
        self._hdu = None
