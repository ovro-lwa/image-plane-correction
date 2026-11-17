"""Tests for the Phase 3 dewarp map vector-field helpers / smoke serve."""

from __future__ import annotations

import socket
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction.live_map import load_tiptilt_npz  # noqa: E402
from image_plane_correction.viz.dewarp_map_app import (  # noqa: E402
    DewarpMapApp,
    _ARROW_MAX_ARCMIN,
    _ARROW_MAX_DISPLAY_DEG,
    _ARROW_REF_ARCMIN,
    _MOLL_X_MAX,
    _arrow_max_display_length,
    _display_uv_from_tip_tilt,
    _grid_spacing_deg,
    build_vectorfield,
    create_app,
    downsample_tiptilt_uv,
    radec_to_mollweide,
    sample_spacing_deg,
    sample_tiptilt_equal_area,
    wrap_lon_offset_deg,
)


def _write_fixture_npz(path: Path, *, scale: float = 1.0) -> None:
    az = np.linspace(0.0, 90.0, 10)
    el = np.linspace(20.0, 80.0, 7)
    az_grid, el_grid = np.meshgrid(az, el, indexing="xy")
    dra = scale * (az_grid - 45.0)
    ddec = scale * (el_grid - 50.0)
    np.savez(
        path,
        az_deg=az,
        el_deg=el,
        dra_arcsec=dra,
        ddec_arcsec=ddec,
        obstime_unix=np.array(1_700_000_000.0),
    )


class TestVectorFieldHelpers(unittest.TestCase):
    def test_radec_to_mollweide_center_and_ra_left(self):
        # Default: RA=0 at plot centre; increasing RA moves left (negative x).
        x0, y0 = radec_to_mollweide(0.0, 0.0, central_ra_deg=0.0)
        self.assertAlmostEqual(float(x0), 0.0, places=6)
        self.assertAlmostEqual(float(y0), 0.0, places=6)
        x_east, _ = radec_to_mollweide(10.0, 0.0, central_ra_deg=0.0)
        x_west, _ = radec_to_mollweide(350.0, 0.0, central_ra_deg=0.0)
        self.assertLess(float(x_east), 0.0)
        self.assertGreater(float(x_west), 0.0)
        # Equator antipode near the limb.
        x_anti, y_anti = radec_to_mollweide(180.0, 0.0, central_ra_deg=0.0)
        self.assertAlmostEqual(float(y_anti), 0.0, places=5)
        self.assertAlmostEqual(abs(float(x_anti)), _MOLL_X_MAX, places=4)

    def test_wrap_lon(self):
        wrapped = wrap_lon_offset_deg(np.array([350.0, 10.0]), 0.0)
        self.assertAlmostEqual(float(wrapped[0]), -10.0, places=6)
        self.assertAlmostEqual(float(wrapped[1]), 10.0, places=6)

    def test_display_uv_scaled_to_10deg_max(self):
        az = np.array([0.0, 90.0, 180.0])
        el = np.array([45.0, 45.0, 45.0])
        # 5' / 10' / 20' — last one clips at _ARROW_MAX_ARCMIN for length.
        dra = np.array([5.0, 10.0, 20.0]) * 60.0  # arcsec
        ddec = np.array([0.0, 0.0, 0.0])
        mag = np.hypot(dra, ddec)
        arrow_max = _arrow_max_display_length(_ARROW_MAX_DISPLAY_DEG)
        _x, _y, u, v, mag_am = _display_uv_from_tip_tilt(
            az,
            el,
            dra,
            ddec,
            mag,
            obstime_unix=1_700_000_000.0,
            central_ra_deg=0.0,
            arrow_max_len=arrow_max,
        )
        disp = np.hypot(u, v)
        self.assertTrue(np.all(np.isfinite(mag_am)))
        self.assertAlmostEqual(float(mag_am[0]), 5.0, places=5)
        # 5' → half length when ref=10'; 10'+ → full 10°-on-sky display length.
        self.assertAlmostEqual(float(disp[0]), 0.5 * arrow_max, places=5)
        self.assertAlmostEqual(float(disp[1]), arrow_max, places=5)
        self.assertLessEqual(float(np.max(disp)), arrow_max * 1.01)
        cell = (2.0 * np.sqrt(2.0) / np.pi) * np.deg2rad(_ARROW_MAX_DISPLAY_DEG)
        self.assertAlmostEqual(arrow_max, cell, places=6)

    def test_grid_spacing_and_arrow_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warp_55MHz.npz"
            _write_fixture_npz(path)
            tip = load_tiptilt_npz(path)
            spacing = _grid_spacing_deg(tip, stride=2)
            self.assertAlmostEqual(spacing, sample_spacing_deg(2), places=6)
            self.assertAlmostEqual(spacing, 5.0, places=6)
            maxlen = _arrow_max_display_length()
            self.assertGreater(maxlen, 0.0)
            self.assertEqual(_ARROW_MAX_DISPLAY_DEG, 20.0)
            self.assertLessEqual(_ARROW_REF_ARCMIN, _ARROW_MAX_ARCMIN)

    def test_equal_area_sample_and_build(self):
        pytest.importorskip("holoviews")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warp_55MHz.npz"
            _write_fixture_npz(path)
            tip = load_tiptilt_npz(path)
            az, el, dra, ddec, mag = sample_tiptilt_equal_area(tip, spacing_deg=10.0)
            self.assertEqual(az.ndim, 1)
            self.assertEqual(az.size, dra.size)
            self.assertTrue(np.all(np.isfinite(mag)))
            # Fixture only covers a sky patch; still expect some hits.
            self.assertGreater(az.size, 0)
            # Legacy az/el stride path still available.
            az2, _, _, _, _ = downsample_tiptilt_uv(tip, stride=2)
            self.assertGreater(az2.size, 0)
            overlay = build_vectorfield(tip, stride=4, central_ra_deg=0.0)
            field = overlay.VectorField.I
            self.assertEqual(field.kdims[0].name, "moll_x")
            self.assertEqual(field.kdims[1].name, "moll_y")
            self.assertGreater(len(field), 0)
            mag_disp = field.dimension_values("Magnitude")
            self.assertGreater(float(np.nanmax(mag_disp)), 0.0)
            hover = overlay.Points.I
            shift = hover.dimension_values("shift_arcmin")
            self.assertGreater(float(np.nanmax(shift)), 0.0)
            self.assertTrue(any("Labels" in str(k) for k in overlay.keys()))

    def test_resolve_path_from_warp_dir_subband(self):
        pytest.importorskip("panel")
        pytest.importorskip("holoviews")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "warp_82MHz.npz"
            _write_fixture_npz(path)
            app = create_app(warp_dir=root, frequency_mhz=82, poll_seconds=60.0, stride=4)
            self.assertEqual(app.resolve_path(), path)
            self.assertIsNotNone(app._tip)
            self.assertIn("subband", app.header.lower())
            self.assertIn("82", app.header)

    def test_create_app_parses_subband_from_warp_path(self):
        pytest.importorskip("panel")
        pytest.importorskip("holoviews")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warp_41MHz.npz"
            _write_fixture_npz(path)
            app = create_app(warp_path=path, warp_dir=None, poll_seconds=60.0, stride=4)
            self.assertEqual(app.frequency_mhz, 41)
            self.assertEqual(app.resolve_path(), path)

    def test_app_reloads_on_mtime(self):
        pytest.importorskip("panel")
        pytest.importorskip("holoviews")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warp_55MHz.npz"
            _write_fixture_npz(path, scale=1.0)
            app = DewarpMapApp(
                warp_path=str(path),
                frequency_mhz=55,
                poll_seconds=30.0,
                stride=1,
            )
            self.assertIsNotNone(app._tip)
            first_mtime = app._loaded_mtime
            first_rev = app.revision
            self.assertFalse(app.reload(force=False))

            time.sleep(0.05)
            _write_fixture_npz(path, scale=2.0)
            path.touch()
            self.assertTrue(app.reload(force=False))
            self.assertNotEqual(app._loaded_mtime, first_mtime)
            app.refresh_beams(force=True)
            app._compose_plot()
            self.assertGreater(app.revision, first_rev)
            self.assertIn("obstime", app.header)
            self.assertIn("BMAJ", app.header)
            self.assertIsNotNone(app._plot_obj)
            layout = app.panel()
            self.assertIsNotNone(layout)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.mark.network
@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_dewarp_map_layout_serves_http():
    """Optional smoke: Panel can serve the dewarp layout over HTTP."""
    pn = pytest.importorskip("panel")
    pytest.importorskip("holoviews")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "warp_55MHz.npz"
        _write_fixture_npz(path)
        app = DewarpMapApp(
            warp_path=str(path),
            frequency_mhz=55,
            poll_seconds=60.0,
            stride=2,
        )
        layout = app.panel()
        port = _free_port()
        server = pn.serve(
            {"dewarp": layout},
            port=port,
            address="127.0.0.1",
            show=False,
            threaded=True,
            verbose=False,
        )
        try:
            deadline = time.monotonic() + 15.0
            last_error: Exception | None = None
            response = None
            body = b""
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/dewarp",
                        timeout=2.0,
                    ) as response:
                        body = response.read()
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                    last_error = exc
                    time.sleep(0.2)
            else:
                raise AssertionError(
                    f"Panel server did not respond: {last_error}"
                ) from last_error

            assert response is not None
            assert response.status == 200
            assert len(body) > 500
            assert b"bokeh" in body.lower() or b"panel" in body.lower()
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
