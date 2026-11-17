"""Tests for tip-tilt publish helpers in watch_calcflow_warp."""

from __future__ import annotations

import importlib.util
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

_SCRIPT = REPO_ROOT / "scripts" / "watch_calcflow_warp.py"
_SPEC = importlib.util.spec_from_file_location("watch_calcflow_warp", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
wcw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wcw)


class TestLatestDayDir(unittest.TestCase):
    def test_latest_day_dir_picks_newest_valid_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2024-12-18").mkdir()
            (root / "2024-12-19").mkdir()
            (root / "not-a-date").mkdir()
            (root / "2024-13-01").mkdir()  # invalid calendar date
            (root / "readme.txt").write_text("ignore", encoding="utf-8")
            latest = wcw._latest_day_dir(root)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.name, "2024-12-19")

    def test_latest_day_dir_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(wcw._latest_day_dir(Path(tmp)))

    def test_parse_day_dirname_rejects_invalid(self):
        self.assertIsNone(wcw._parse_day_dirname("2024-13-01"))
        self.assertIsNone(wcw._parse_day_dirname("20241218"))
        parsed = wcw._parse_day_dirname("2024-12-18")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.isoformat(), "2024-12-18")

    def test_latest_fz_only_searches_day_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2024-12-19"
            day.mkdir()
            older = day / "55MHz-I-20241219_010000-image.fits.fz"
            newer = day / "55MHz-I-20241219_020000-image.fits.fz"
            older.write_bytes(b"fz")
            newer.write_bytes(b"fz")
            # Sibling day should not affect selection when calling _latest_fz on day.
            other = root / "2024-12-20"
            other.mkdir()
            (other / "55MHz-I-20241220_030000-image.fits.fz").write_bytes(b"fz")
            self.assertEqual(wcw._latest_fz(day), newer)


class TestPublishTiptilt(unittest.TestCase):
    def test_publish_multiple_frequencies_calls_set_tiptilt_per_key(self):
        calls = []

        def fake_set_tiptilt(path, ttl_sec, frequency_mhz=None, client=None):
            calls.append((path, float(ttl_sec), frequency_mhz))

        tiptilt_mod = types.ModuleType("mnc.tiptilt")
        tiptilt_mod.set_tiptilt = fake_set_tiptilt
        mnc_mod = types.ModuleType("mnc")
        mnc_mod.tiptilt = tiptilt_mod

        with tempfile.TemporaryDirectory() as tmp:
            warp = Path(tmp) / "warp_55MHz.npz"
            warp.write_bytes(b"npz")
            with mock.patch.dict(
                sys.modules, {"mnc": mnc_mod, "mnc.tiptilt": tiptilt_mod}
            ):
                wcw._publish_tiptilt(
                    warp,
                    ttl_sec=123.0,
                    frequencies_mhz=(55, 82, 41),
                )

        abs_path = str(warp.resolve())
        self.assertEqual(
            calls,
            [
                (abs_path, 123.0, 55),
                (abs_path, 123.0, 82),
                (abs_path, 123.0, 41),
            ],
        )

    def test_parser_accepts_publish_all_frequencies(self):
        args = wcw.build_parser().parse_args(
            [
                "--watch-dir",
                "/tmp",
                "--output-dir",
                "/tmp",
                "--warp-filename",
                "warp_55MHz.npz",
                "--publish-all-frequencies",
            ]
        )
        self.assertTrue(args.publish_all_frequencies)


if __name__ == "__main__":
    unittest.main()
