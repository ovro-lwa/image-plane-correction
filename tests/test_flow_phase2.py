import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction import flow


class TestFlowCascade73MHzPhase2(unittest.TestCase):
    def _touch(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def test_phase2_aggregates_results_by_subband(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)

            # Build three subbands with complete 70/73/75 MHz peers.
            files = {}
            for sb in ["18MHz", "23MHz", "27MHz"]:
                for freq in [70, 73, 75]:
                    image = (
                        work_dir / sb / f"{freq}MHz-I-Taper-10min-20241218_030021-image.fits"
                    )
                    psf = work_dir / sb / f"{freq}MHz-I-Taper-10min-20241218_030021-psf.fits"
                    self._touch(image)
                    self._touch(psf)
                    files[(sb, freq)] = str(image)

            qa_by_path = {
                files[("18MHz", 73)]: True,
                files[("18MHz", 75)]: True,
                files[("18MHz", 70)]: True,
                files[("23MHz", 73)]: True,
                files[("23MHz", 75)]: False,  # one failure causes 23MHz=False
                files[("23MHz", 70)]: True,
                files[("27MHz", 73)]: True,
                files[("27MHz", 75)]: True,
                files[("27MHz", 70)]: True,
            }

            def _fake_calcflow(image_fn, **kwargs):
                qa_ok = qa_by_path[image_fn]
                return None, f"ref::{os.path.basename(image_fn)}", None, None, qa_ok

            logger = mock.Mock()
            with mock.patch.object(flow, "calcflow", side_effect=_fake_calcflow):
                out = flow.flow_cascade73MHz_phase2(str(work_dir), logger)

            self.assertTrue(out["18MHz"])
            self.assertFalse(out["23MHz"])
            self.assertTrue(out["27MHz"])
            self.assertFalse(out["82MHz"])  # in fixed subband list but missing from work_dir
            self.assertEqual(len(out), len(flow.PHASE2_SUBBANDS))

            logger.info.assert_called_once_with(
                "Phase2 cascade QA summary n_ok=%s n_fail=%s n_missing=%s",
                2,  # 18MHz, 27MHz
                1,  # 23MHz
                12,  # fixed list has 15 total and only 3 represented in results
            )
            logger.debug.assert_called_once()


if __name__ == "__main__":
    unittest.main()
