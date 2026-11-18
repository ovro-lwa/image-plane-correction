"""Catalog CSV loads are memoized across repeated reference_sources_* calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction import catalogs  # noqa: E402


def _fake_nvss_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rah": [12],
            "ram": [0],
            "ras": [0.0],
            "dd": [45],
            "dm": [0],
            "ds": [0.0],
            "f": [500.0],
        }
    )


def _fake_vlssr_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RA": [180.0],
            "DEC": [45.0],
            "PEAK INT": [200.0],
        }
    )


class TestCatalogCache(unittest.TestCase):
    def tearDown(self) -> None:
        catalogs.clear_catalog_cache()

    def test_nvss_read_csv_once_for_repeated_loads(self) -> None:
        catalogs.clear_catalog_cache()
        path = Path("/tmp/fake_nvss_catalog.dat")
        stamp = (str(path), 1, 10)
        df = _fake_nvss_df()
        with (
            mock.patch.object(catalogs, "_catalog_file_stamp", return_value=stamp),
            mock.patch.object(catalogs.pd, "read_csv", return_value=df) as read_csv,
        ):
            catalogs.reference_sources_nvss(min_flux=100, path=str(path))
            catalogs.reference_sources_nvss(min_flux=200, path=str(path))
            catalogs.reference_sources_nvss(min_flux=100, path=str(path))
        self.assertEqual(read_csv.call_count, 1)

    def test_vlssr_read_csv_once_for_repeated_loads(self) -> None:
        catalogs.clear_catalog_cache()
        path = Path("/tmp/fake_vlssr_catalog.txt")
        stamp = (str(path), 2, 20)
        df = _fake_vlssr_df()
        with (
            mock.patch.object(catalogs, "_catalog_file_stamp", return_value=stamp),
            mock.patch.object(catalogs.pd, "read_csv", return_value=df) as read_csv,
        ):
            catalogs.reference_sources_vlssr(min_flux=10, path=str(path))
            catalogs.reference_sources_vlssr(min_flux=50, path=str(path))
            catalogs.reference_sources_vlssr(min_flux=10, path=str(path))
        self.assertEqual(read_csv.call_count, 1)

    def test_clear_catalog_cache_forces_reload(self) -> None:
        catalogs.clear_catalog_cache()
        path = Path("/tmp/fake_vlssr_catalog.txt")
        stamp = (str(path), 3, 30)
        df = _fake_vlssr_df()
        with (
            mock.patch.object(catalogs, "_catalog_file_stamp", return_value=stamp),
            mock.patch.object(catalogs.pd, "read_csv", return_value=df) as read_csv,
        ):
            catalogs.reference_sources_vlssr(min_flux=10, path=str(path))
            catalogs.clear_catalog_cache()
            catalogs.reference_sources_vlssr(min_flux=10, path=str(path))
        self.assertEqual(read_csv.call_count, 2)


if __name__ == "__main__":
    unittest.main()
