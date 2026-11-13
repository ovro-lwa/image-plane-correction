import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction import util as util_mod  # noqa: E402
from image_plane_correction.util import runqa  # noqa: E402


class TestRunqaIgnoresNan(unittest.TestCase):
    def test_residual_percentiles_ignore_nan_regions(self):
        n = 16
        rng = np.random.default_rng(0)
        image = rng.standard_normal((n, n)).astype(np.float64)
        reference_sky = rng.standard_normal((n, n)).astype(np.float64)
        dewarped = image * 0.95 + 0.01

        reference_sky[:4, :4] = np.nan
        dewarped[:4, :4] = np.nan

        class ZeroFlow:
            offsets = np.zeros((n, n, 2), dtype=np.float64)

        dew64, img64, ref64 = util_mod._qa_arrays_to_numpy_f64(dewarped, image, reference_sky)
        residuals, n_used = util_mod._qa_residual_difference_percentiles(
            dew64, img64, ref64, [5, 50, 95]
        )
        self.assertGreater(n_used, 0)
        self.assertTrue(np.all(np.isfinite(residuals)))

        score = runqa(image, reference_sky, ZeroFlow(), dewarped)
        self.assertIn(score, (0, 1))


if __name__ == "__main__":
    unittest.main()
