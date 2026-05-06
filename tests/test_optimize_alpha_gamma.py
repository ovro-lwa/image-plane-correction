import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_opt_script():
    name = "optimize_alpha_gamma_cli"
    path = SCRIPTS_DIR / "optimize_alpha_gamma.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestOptimizeAggregate(unittest.TestCase):
    def test_aggregate_recommends_higher_median_score(self):
        mod = _load_opt_script()
        rows = [
            {
                "alpha": 1.0,
                "gamma": 10.0,
                "metrics_structure_score": 0.5,
                "runqa_score": 1,
                "shift_p50": 1.0,
                "error": None,
            },
            {
                "alpha": 2.0,
                "gamma": 10.0,
                "metrics_structure_score": 0.9,
                "runqa_score": 1,
                "shift_p50": 2.0,
                "error": None,
            },
        ]
        out = mod._aggregate(rows, bootstrap_samples=0, rng=np.random.default_rng(0), min_qa_rate=0.0)
        self.assertIsNotNone(out["recommended"])
        assert out["recommended"] is not None
        self.assertEqual(out["recommended"]["alpha"], 2.0)

    def test_aggregate_prefers_composite_when_finite(self):
        mod = _load_opt_script()
        rows = [
            {
                "alpha": 1.0,
                "gamma": 10.0,
                "metrics_structure_score": 0.8,
                "metrics_curl_to_div_ratio_band": 2.0,
                "composite_objective": 0.9,
                "runqa_score": 1,
                "qa_passed": True,
                "shift_p50": 1.0,
                "error": None,
            },
            {
                "alpha": 2.0,
                "gamma": 10.0,
                "metrics_structure_score": 0.85,
                "metrics_curl_to_div_ratio_band": 0.5,
                "composite_objective": 1.2,
                "runqa_score": 1,
                "qa_passed": True,
                "shift_p50": 2.0,
                "error": None,
            },
        ]
        out = mod._aggregate(rows, bootstrap_samples=0, rng=np.random.default_rng(0), min_qa_rate=0.0)
        self.assertTrue(out["ranked_by_composite"])
        assert out["recommended"] is not None
        self.assertEqual(out["recommended"]["alpha"], 2.0)

    def test_rows_for_aggregate_prefers_refine_best(self):
        mod = _load_opt_script()
        rows = [
            {
                "image": "a.fits",
                "alpha": 1.0,
                "gamma": 10.0,
                "phase": "grid",
                "error": None,
                "metrics_structure_score": 0.5,
            },
            {
                "image": "a.fits",
                "alpha": 1.0,
                "gamma": 10.0,
                "phase": "refine_best",
                "error": None,
                "metrics_structure_score": 0.9,
            },
        ]
        deduped = mod._rows_for_aggregate(rows)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["phase"], "refine_best")

    def test_evaluate_one_mocked_smoke(self):
        mod = _load_opt_script()

        class FakeFlow:
            offsets = np.zeros((32, 32, 2), dtype=np.float64)

        def fake_calcflow(**kwargs):
            img = np.zeros((32, 32), dtype=np.float64)
            return img, img, FakeFlow(), img, True

        def fake_fits_image(path):
            from astropy.io import fits
            from astropy.wcs import WCS

            h = fits.Header()
            h["NAXIS"] = 2
            h["NAXIS1"] = 32
            h["NAXIS2"] = 32
            h["CTYPE1"] = "RA---TAN"
            h["CTYPE2"] = "DEC--TAN"
            h["CRVAL1"] = 180.0
            h["CRVAL2"] = 45.0
            h["CRPIX1"] = 16.0
            h["CRPIX2"] = 16.0
            h["CDELT1"] = -0.05
            h["CDELT2"] = 0.05
            h["CUNIT1"] = "deg"
            h["CUNIT2"] = "deg"
            return np.zeros((32, 32), dtype=np.float64), WCS(h)

        spec = mod.ImageSpec(image="/tmp/fake.fits", psf="/tmp/fake.psf", reference_sky_fn=None)

        with mock.patch.object(mod, "calcflow", side_effect=fake_calcflow), mock.patch.object(
            mod, "fits_image", side_effect=fake_fits_image
        ), mock.patch.object(mod, "horizon_r_normalized", return_value=0.7):
            row = mod.evaluate_one(
                spec,
                1.3,
                150.0,
                cleaned=False,
                band_deg=(20.0, 100.0),
                structure_mask="disk",
                horizon_elevation_deg=10.0,
                catalog="VLSSR",
                catalog_path="/home/claw/vlssr_radecpeak_unresolved.txt",
                preprocess_weight=1.5,
                scale_factor=0.7,
                use_best_pb_model=False,
                bright_source_flux_qa=False,
                bright_source_flux_qa_count=10,
                qa=True,
                quiet=True,
                w_struct=1.0,
                w_qa=1.0,
                soft_qa=False,
            )

        self.assertIsNone(row["error"])
        self.assertIn("metrics_structure_score", row)
        self.assertEqual(row["runqa_score"], 1)
        self.assertEqual(row["qa_passed"], True)
        self.assertIsNotNone(row["composite_objective"])
        assert row["composite_objective"] is not None
        self.assertGreater(row["composite_objective"], 0.0)


class TestOptimizeCLI(unittest.TestCase):
    def test_main_writes_json_expected_keys(self):
        mod = _load_opt_script()

        class FakeFlow:
            offsets = np.zeros((16, 16, 2), dtype=np.float64)

        def fake_calcflow(**kwargs):
            img = np.zeros((16, 16), dtype=np.float64)
            return img, img, FakeFlow(), img, True

        def fake_fits_image(path):
            from astropy.io import fits
            from astropy.wcs import WCS

            h = fits.Header()
            h["NAXIS"] = 2
            h["NAXIS1"] = 16
            h["NAXIS2"] = 16
            h["CTYPE1"] = "RA---TAN"
            h["CTYPE2"] = "DEC--TAN"
            h["CRVAL1"] = 180.0
            h["CRVAL2"] = 45.0
            h["CRPIX1"] = 8.0
            h["CRPIX2"] = 8.0
            h["CDELT1"] = -0.05
            h["CDELT2"] = 0.05
            h["CUNIT1"] = "deg"
            h["CUNIT2"] = "deg"
            return np.zeros((16, 16), dtype=np.float64), WCS(h)

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.json"
            argv = [
                "--images",
                "dummy.fits",
                "--psf",
                "dummy.psf",
                "--alphas",
                "1.0",
                "--gammas",
                "10,20",
                "--bootstrap",
                "0",
                "--structure-mask",
                "none",
                "--output-json",
                str(out_path),
            ]
            with mock.patch.object(mod, "calcflow", side_effect=fake_calcflow), mock.patch.object(
                mod, "fits_image", side_effect=fake_fits_image
            ), mock.patch.object(mod, "horizon_r_normalized", return_value=0.7):
                mod.main(argv)

            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], mod.SCHEMA_VERSION)
            self.assertEqual(len(data["rows"]), 2)
            self.assertIn("aggregate", data)
            self.assertIn("metrics_structure_score", data["rows"][0])
            self.assertIn("composite_objective", data["rows"][0])

    def test_search_mode_small_grid(self):
        mod = _load_opt_script()

        class FakeFlow:
            offsets = np.zeros((16, 16, 2), dtype=np.float64)

        def fake_calcflow(**kwargs):
            img = np.zeros((16, 16), dtype=np.float64)
            return img, img, FakeFlow(), img, True

        def fake_fits_image(path):
            from astropy.io import fits
            from astropy.wcs import WCS

            h = fits.Header()
            h["NAXIS"] = 2
            h["NAXIS1"] = 16
            h["NAXIS2"] = 16
            h["CTYPE1"] = "RA---TAN"
            h["CTYPE2"] = "DEC--TAN"
            h["CRVAL1"] = 180.0
            h["CRVAL2"] = 45.0
            h["CRPIX1"] = 8.0
            h["CRPIX2"] = 8.0
            h["CDELT1"] = -0.05
            h["CDELT2"] = 0.05
            h["CUNIT1"] = "deg"
            h["CUNIT2"] = "deg"
            return np.zeros((16, 16), dtype=np.float64), WCS(h)

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "search.json"
            argv = [
                "--search",
                "--alpha-min",
                "1",
                "--alpha-max",
                "2",
                "--alpha-steps",
                "2",
                "--gamma-min",
                "10",
                "--gamma-max",
                "20",
                "--gamma-steps",
                "2",
                "--images",
                "dummy.fits",
                "--psf",
                "dummy.psf",
                "--bootstrap",
                "0",
                "--structure-mask",
                "none",
                "--output-json",
                str(out_path),
            ]
            with mock.patch.object(mod, "calcflow", side_effect=fake_calcflow), mock.patch.object(
                mod, "fits_image", side_effect=fake_fits_image
            ), mock.patch.object(mod, "horizon_r_normalized", return_value=0.7):
                mod.main(argv)

            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(data["params"]["search"])
            self.assertEqual(data["params"]["grid_size"], 4)
            self.assertEqual(len(data["rows"]), 4)


if __name__ == "__main__":
    unittest.main()
