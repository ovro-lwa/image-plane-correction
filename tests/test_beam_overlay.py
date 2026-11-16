"""Tests for Phase 4 beam pointing overlays / warp-at-beam rows."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction.live_map import load_tiptilt_npz  # noqa: E402
from image_plane_correction.viz.beam_overlay import (  # noqa: E402
    build_beam_overlay,
    collect_beam_rows,
    load_demo_pointings,
)
from image_plane_correction.viz.dewarp_map_app import create_app  # noqa: E402

# Synthetic pointings for unit tests only (not the empty ops demo fixture).
_TEST_POINTINGS = {
    "beams": {
        "1": {
            "timestamp": 1890000000.0,
            "unit": "deg",
            "value": {"nominal": {"az_deg": 120.0, "el_deg": 45.0}},
        },
        "3": {
            "timestamp": 1890000000.0,
            "unit": "deg",
            "value": {
                "nominal": {"az_deg": 200.0, "el_deg": 55.0},
                "commanded_by_frequency_mhz": {
                    "55": {"az_deg": 201.5, "el_deg": 54.2}
                },
            },
        },
        "5": {
            "timestamp": 1000.0,
            "unit": "deg",
            "value": {"nominal": {"az_deg": 30.0, "el_deg": 70.0}},
        },
    }
}


def _write_test_pointings(path: Path) -> None:
    path.write_text(json.dumps(_TEST_POINTINGS), encoding="utf-8")


def _write_fixture_npz(path: Path) -> None:
    az = np.linspace(0.0, 360.0, 37)
    el = np.linspace(0.0, 90.0, 19)
    az_grid, el_grid = np.meshgrid(az, el, indexing="xy")
    dra = 0.1 * (az_grid - 180.0)
    ddec = 0.2 * (el_grid - 45.0)
    np.savez(
        path,
        az_deg=az,
        el_deg=el,
        dra_arcsec=dra,
        ddec_arcsec=ddec,
        obstime_unix=np.array(1_700_000_000.0),
    )


class TestBeamOverlay(unittest.TestCase):
    def test_load_demo_and_collect_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            pointing_path = Path(tmp) / "pointings.json"
            _write_test_pointings(pointing_path)
            demo = load_demo_pointings(pointing_path)
            self.assertEqual(set(demo), {1, 3, 5})

            path = Path(tmp) / "warp_55MHz.npz"
            _write_fixture_npz(path)
            tip = load_tiptilt_npz(path)
            # Freeze "now" so beam 5 (timestamp=1000) is stale, 1/3 are fresh.
            now = 1890000010.0
            rows = collect_beam_rows(
                frequency_mhz=55,
                tip=tip,
                max_age_sec=120.0,
                demo_pointings=demo,
                now=now,
            )
        by_beam = {r.beam: r for r in rows}
        self.assertEqual(by_beam[1].status, "ok")
        self.assertFalse(by_beam[1].uses_commanded)
        self.assertEqual(by_beam[1].display_az_deg, 120.0)
        self.assertEqual(by_beam[3].status, "ok")
        self.assertTrue(by_beam[3].uses_commanded)
        self.assertEqual(by_beam[3].display_az_deg, 201.5)
        self.assertEqual(by_beam[3].nominal_az_deg, 200.0)
        self.assertIsNotNone(by_beam[3].shift_arcsec)
        self.assertEqual(by_beam[5].status, "stale")

    def test_missing_pointing_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            pointing_path = Path(tmp) / "pointings.json"
            _write_test_pointings(pointing_path)
            demo = load_demo_pointings(pointing_path)
        rows = collect_beam_rows(
            frequency_mhz=55,
            tip=None,
            demo_pointings=demo,
            demo_beam_numbers=[1, 99],
            now=1890000010.0,
        )
        by_beam = {r.beam: r for r in rows}
        self.assertEqual(by_beam[1].status, "ok")
        self.assertEqual(by_beam[99].status, "no_pointing")
        self.assertIsNone(by_beam[99].display_az_deg)

    def test_discover_failure_falls_back_to_pointing_keys(self):
        def boom(*, type, age_limit):
            raise RuntimeError("etcd too large")

        rows = collect_beam_rows(
            frequency_mhz=55,
            tip=None,
            max_age_sec=float("inf"),
            discover_recorders=boom,
            list_pointing_beams=lambda: [1, 3],
            read_pointing=lambda beam: None,
            now=1890000010.0,
        )
        self.assertEqual([r.beam for r in rows], [1, 3])
        self.assertTrue(all(r.status == "no_pointing" for r in rows))

    def test_build_overlay_and_app_demo(self):
        pytest.importorskip("holoviews")
        pytest.importorskip("panel")
        with tempfile.TemporaryDirectory() as tmp:
            pointing_path = Path(tmp) / "pointings.json"
            _write_test_pointings(pointing_path)
            demo = load_demo_pointings(pointing_path)
            path = Path(tmp) / "warp_55MHz.npz"
            _write_fixture_npz(path)
            tip = load_tiptilt_npz(path)
            rows = collect_beam_rows(
                frequency_mhz=55,
                tip=tip,
                demo_pointings=demo,
                now=1890000010.0,
            )
            overlay = build_beam_overlay(
                rows,
                obstime_unix=float(tip.obstime_unix),
                central_ra_deg=0.0,
            )
            self.assertIsNotNone(overlay)

            fresh = load_demo_pointings(pointing_path)
            for beam, rec in list(fresh.items()):
                fresh[beam] = type(rec)(
                    beam=rec.beam,
                    timestamp=time.time() if beam != 5 else time.time() - 1000.0,
                    nominal_az_deg=rec.nominal_az_deg,
                    nominal_el_deg=rec.nominal_el_deg,
                    commanded_by_frequency_mhz=dict(rec.commanded_by_frequency_mhz),
                    unit=rec.unit,
                )
            app = create_app(
                warp_path=path,
                frequency_mhz=55,
                poll_seconds=60.0,
                demo_pointings_path=pointing_path,
            )
            app._demo_pointings = fresh
            self.assertTrue(app.refresh_beams(force=True))
            app._compose_plot()
            self.assertIsNotNone(app._plot_obj)
            self.assertIn("dr1", app.beam_table_html)
            self.assertIn("dr3", app.beam_table_html)
            layout = app.panel()
            self.assertIsNotNone(layout)


if __name__ == "__main__":
    unittest.main()
