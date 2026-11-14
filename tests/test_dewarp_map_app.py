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
    _ZEA_R_HORIZON,
    azel_to_zea,
    build_vectorfield,
    downsample_tiptilt_uv,
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
    def test_azel_to_zea_zenith_and_horizon(self):
        # Zenith → origin; north horizon → +y at √2; east horizon → +x at √2.
        xz, yz = azel_to_zea(0.0, 90.0, central_az_deg=0.0)
        self.assertAlmostEqual(float(xz), 0.0, places=6)
        self.assertAlmostEqual(float(yz), 0.0, places=6)
        xn, yn = azel_to_zea(0.0, 0.0, central_az_deg=0.0)
        self.assertAlmostEqual(float(xn), 0.0, places=6)
        self.assertAlmostEqual(float(yn), _ZEA_R_HORIZON, places=5)
        xe, ye = azel_to_zea(90.0, 0.0, central_az_deg=0.0)
        self.assertAlmostEqual(float(xe), _ZEA_R_HORIZON, places=5)
        self.assertAlmostEqual(float(ye), 0.0, places=5)

    def test_downsample_and_build(self):
        pytest.importorskip("holoviews")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warp_55MHz.npz"
            _write_fixture_npz(path)
            tip = load_tiptilt_npz(path)
            az, el, dra, ddec, mag = downsample_tiptilt_uv(tip, stride=2)
            self.assertEqual(az.ndim, 1)
            self.assertEqual(az.size, dra.size)
            self.assertTrue(np.all(np.isfinite(mag)))
            self.assertGreater(az.size, 0)
            overlay = build_vectorfield(tip, stride=2)
            # Overlay = outline + graticule curves + VectorField
            field = overlay.VectorField.I
            self.assertEqual(field.kdims[0].name, "zea_x")
            self.assertEqual(field.kdims[1].name, "zea_y")
            self.assertGreater(len(field), 0)

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
            # Ensure mtime advances on coarse filesystems.
            path.touch()
            self.assertTrue(app.reload(force=False))
            self.assertNotEqual(app._loaded_mtime, first_mtime)
            app.refresh_beams(force=True)
            app._compose_plot()
            self.assertGreater(app.revision, first_rev)
            self.assertIn("obstime", app.header)
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
