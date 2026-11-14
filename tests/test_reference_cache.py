"""ReferenceSkyCache reuses HDUs across sequential calcflow calls."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from image_plane_correction.flow import calcflow  # noqa: E402
from image_plane_correction.reference_cache import (  # noqa: E402
    ReferenceSkyCache,
    obs_time_from_header,
)
import image_plane_correction.catalogs as catalogs  # noqa: E402


def _wcs_tan(n: int, cdelt_deg: float = 0.1) -> WCS:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = n
    h["NAXIS2"] = n
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = 180.0
    h["CRVAL2"] = 45.0
    h["CRPIX1"] = (n + 1) / 2.0
    h["CRPIX2"] = (n + 1) / 2.0
    h["CDELT1"] = -cdelt_deg
    h["CDELT2"] = cdelt_deg
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    return WCS(h)


def _image_header(n: int, *, date_obs: str = "2026-11-13T05:52:45.000") -> fits.Header:
    hdr = _wcs_tan(n).to_header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = n
    hdr["NAXIS2"] = n
    hdr["BMAJ"] = 0.2
    hdr["BMIN"] = 0.15
    hdr["BPA"] = 0.0
    hdr["DATE-OBS"] = date_obs
    return hdr


class TestReferenceSkyCache(unittest.TestCase):
    def test_second_calcflow_skips_theoretical_sky_same_obstime(self) -> None:
        n = 32
        hdr = _image_header(n)
        cache = ReferenceSkyCache()
        calls = {"n": 0}

        def fake_theory(*_a, **_k):
            calls["n"] += 1
            out = np.zeros((n, n), dtype=np.float32)
            out[n // 2, n // 2] = 1.0
            return out

        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in range(2):
                p = Path(td) / f"img{i}.fits"
                img = np.random.default_rng(i).standard_normal((n, n)).astype(np.float32)
                fits.writeto(p, img, hdr, overwrite=True)
                paths.append(str(p))

            with mock.patch.object(
                catalogs, "theoretical_sky_beam_function", side_effect=fake_theory
            ):
                for path in paths:
                    key = ReferenceSkyCache.make_key(
                        header=fits.getheader(path),
                        target_size=None,
                    )
                    cached = cache.get(key)
                    _img, ref, flow, dew, ok = calcflow(
                        path,
                        reference_sky=cached,
                        cleaned=True,
                        qa=False,
                        dewarp=False,
                        catalog_path="/dev/null",
                    )
                    if cached is None:
                        cache.store(key, ref)
                    self.assertTrue(ok)
                    self.assertEqual(tuple(np.asarray(flow.offsets).shape), (n, n, 2))
                    self.assertIsNone(dew)

        self.assertEqual(calls["n"], 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_key_changes_with_obstime(self) -> None:
        n = 16
        h1 = _image_header(n, date_obs="2026-11-13T05:52:45.000")
        h2 = _image_header(n, date_obs="2026-11-13T05:53:45.000")
        k1 = ReferenceSkyCache.make_key(header=h1, target_size=512)
        k2 = ReferenceSkyCache.make_key(header=h2, target_size=512)
        self.assertNotEqual(k1, k2)
        self.assertNotEqual(obs_time_from_header(h1), obs_time_from_header(h2))

    def test_key_changes_with_target_size(self) -> None:
        h = _image_header(16)
        k1 = ReferenceSkyCache.make_key(header=h, target_size=512)
        k2 = ReferenceSkyCache.make_key(header=h, target_size=1024)
        self.assertNotEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
