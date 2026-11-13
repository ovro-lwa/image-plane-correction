"""Tests for pixel-flow → tip-tilt NPZ conversion."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction.tiptilt_map import (  # noqa: E402
    frequency_mhz_from_filename,
    obstime_unix_from_header,
    pixel_flow_to_tiptilt_arrays,
    write_tiptilt_npz,
)


def _wcs_sin(n: int, cdelt_deg: float = 0.5) -> WCS:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = n
    h["NAXIS2"] = n
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CRVAL1"] = 180.0
    h["CRVAL2"] = 37.0
    h["CRPIX1"] = (n + 1) / 2.0
    h["CRPIX2"] = (n + 1) / 2.0
    h["CDELT1"] = -cdelt_deg
    h["CDELT2"] = cdelt_deg
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    h["RADESYS"] = "FK5"
    h["EQUINOX"] = 2000.0
    return WCS(h)


class TestTiptiltMap(unittest.TestCase):
    def test_frequency_from_filename(self):
        self.assertEqual(frequency_mhz_from_filename("warp_55MHz.npz"), 55)
        self.assertEqual(frequency_mhz_from_filename("/tmp/flow_82MHz_map.npz"), 82)
        with self.assertRaises(ValueError):
            frequency_mhz_from_filename("warp.npz")

    def test_obstime_from_header(self):
        h = fits.Header()
        h["DATE-OBS"] = "2026-11-13T12:00:00"
        unix = obstime_unix_from_header(h)
        self.assertAlmostEqual(unix, Time("2026-11-13T12:00:00", scale="utc").unix)

    def test_constant_pixel_offset_roundtrip_npz_schema(self):
        n = 64
        imwcs = _wcs_sin(n, cdelt_deg=0.5)
        # Constant +2 px in x (column / RA-ish for this WCS).
        offsets = np.zeros((n, n, 2), dtype=np.float64)
        offsets[:, :, 0] = 2.0

        location = EarthLocation(lat=37.23335 * u.deg, lon=-118.28167 * u.deg, height=1222 * u.m)
        obstime = Time("2024-01-01T08:00:00", scale="utc")
        obstime_unix = float(obstime.unix)

        cy = cx = (n - 1) / 2.0
        sky0 = imwcs.pixel_to_world(cx, cy)
        sky1 = imwcs.pixel_to_world(cx + 2.0, cy)
        lon, lat = sky0.spherical_offsets_to(sky1)
        expected_dra = float(lon.to_value("arcsec"))
        expected_ddec = float(lat.to_value("arcsec"))

        # Grid centered on the image CRVAL's AltAz so samples land on-sky in the SIN FOV.
        aa = sky0.transform_to(AltAz(obstime=obstime, location=location))
        az0 = float(aa.az.deg)
        el0 = float(aa.alt.deg)
        self.assertGreater(el0, 5.0)
        az_deg = np.array([az0 - 2.0, az0, az0 + 2.0], dtype=np.float64)
        el_deg = np.array([el0 - 2.0, el0, min(el0 + 2.0, 90.0)], dtype=np.float64)

        arrays = pixel_flow_to_tiptilt_arrays(
            offsets,
            imwcs,
            obstime_unix,
            location=location,
            az_deg=az_deg,
            el_deg=el_deg,
        )
        self.assertEqual(arrays["dra_arcsec"].shape, (3, 3))
        self.assertTrue(np.isfinite(arrays["dra_arcsec"][1, 1]))
        self.assertAlmostEqual(arrays["dra_arcsec"][1, 1], expected_dra, delta=1.0)
        self.assertAlmostEqual(arrays["ddec_arcsec"][1, 1], expected_ddec, delta=1.0)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "warp_55MHz.npz"
            write_tiptilt_npz(out, arrays)
            loaded = np.load(out)
            self.assertEqual(loaded["dra_arcsec"].shape, (3, 3))
            self.assertAlmostEqual(float(loaded["obstime_unix"]), obstime_unix)
            loaded.close()


if __name__ == "__main__":
    unittest.main()
