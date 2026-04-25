import os
import sys
import tempfile
import types
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

            all_subbands = ["SB001", "SB002", "SB003", "SB004"]
            cfg_mod = types.SimpleNamespace(ALL_SUBBANDS=all_subbands)

            # Build three subbands with complete 70/73/75 MHz peers.
            files = {}
            for sb in ["SB001", "SB002", "SB003"]:
                for freq in [70, 73, 75]:
                    image = (
                        work_dir / sb / f"{freq}MHz-I-Taper-10min-20241218_030021-image.fits"
                    )
                    psf = work_dir / sb / f"{freq}MHz-I-Taper-10min-20241218_030021-psf.fits"
                    self._touch(image)
                    self._touch(psf)
                    files[(sb, freq)] = str(image)

            qa_by_path = {
                files[("SB001", 73)]: True,
                files[("SB001", 75)]: True,
                files[("SB001", 70)]: True,
                files[("SB002", 73)]: True,
                files[("SB002", 75)]: False,  # one failure causes SB002=False
                files[("SB002", 70)]: True,
                files[("SB003", 73)]: True,
                files[("SB003", 75)]: True,
                files[("SB003", 70)]: True,
            }

            def _fake_calcflow(image_fn, **kwargs):
                qa_ok = qa_by_path[image_fn]
                return None, f"ref::{os.path.basename(image_fn)}", None, None, qa_ok

            logger = mock.Mock()
            with mock.patch.dict(sys.modules, {"cfg": cfg_mod}, clear=False):
                with mock.patch.object(flow, "calcflow", side_effect=_fake_calcflow):
                    out = flow.flow_cascade73MHz_phase2(str(work_dir), logger)

            self.assertEqual(
                out,
                {
                    "SB001": True,
                    "SB002": False,
                    "SB003": True,
                    "SB004": False,  # present in cfg.ALL_SUBBANDS but missing from work_dir
                },
            )

            logger.info.assert_called_once_with(
                "Phase2 cascade QA summary n_ok=%s n_fail=%s n_missing=%s",
                2,  # SB001, SB003
                1,  # SB002
                1,  # SB004 missing from results
            )
            logger.debug.assert_called_once()


if __name__ == "__main__":
    unittest.main()
