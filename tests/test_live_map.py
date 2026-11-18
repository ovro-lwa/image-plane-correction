"""Tests for tip-tilt NPZ loaders and beam pointing accessors."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction.live_map import (  # noqa: E402
    beam_number_from_recorder_id,
    display_az_el,
    is_pointing_stale,
    list_active_beam_numbers,
    list_beam_numbers_from_pointing_keys,
    load_tiptilt_npz,
    parse_beam_pointing_payload,
    pointing_age_sec,
    read_beam_pointing,
)


def _write_fixture_npz(path: Path) -> None:
    az = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    el = np.array([40.0, 50.0], dtype=np.float64)
    # shape (n_el, n_az): dra = az, ddec = el (easy to sample)
    az_grid, el_grid = np.meshgrid(az, el, indexing="xy")
    # meshgrid with indexing xy → az_grid shape (n_el, n_az) when el rows, az cols
    # Actually np.meshgrid(..., indexing="xy") with az (3,), el (2,) yields
    # az_grid (2,3), el_grid (2,3) — perfect.
    dra = az_grid.copy()
    ddec = el_grid.copy()
    np.savez(
        path,
        az_deg=az,
        el_deg=el,
        dra_arcsec=dra,
        ddec_arcsec=ddec,
        obstime_unix=np.array(1_700_000_000.0),
    )


class _FakeMeta:
    def __init__(self, key: str):
        self.key = key.encode() if isinstance(key, str) else key


class FakeEtcd:
    def __init__(self, store):
        self.store = store
        self.closed = False

    def get(self, key):
        value = self.store.get(key)
        return (value, None)

    def get_prefix(self, prefix):
        for key, value in self.store.items():
            if str(key).startswith(prefix):
                yield value, _FakeMeta(str(key))

    def close(self):
        self.closed = True


class TestTipTiltMapLoad(unittest.TestCase):
    def test_sample_at_known_grid_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warp_55MHz.npz"
            _write_fixture_npz(path)
            tip = load_tiptilt_npz(path)
            self.assertEqual(tip.obstime_unix, 1_700_000_000.0)
            dra, ddec, outside = tip.sample(20.0, 50.0)
            self.assertFalse(outside)
            self.assertAlmostEqual(dra, 20.0)
            self.assertAlmostEqual(ddec, 50.0)
            mid_dra, mid_ddec, mid_out = tip.sample(15.0, 45.0)
            self.assertFalse(mid_out)
            self.assertAlmostEqual(mid_dra, 15.0)
            self.assertAlmostEqual(mid_ddec, 45.0)
            _, _, out = tip.sample(0.0, 0.0)
            self.assertTrue(out)
            self.assertTrue(np.isnan(tip.magnitude_arcsec(0.0, 0.0)))
            self.assertAlmostEqual(tip.magnitude_arcsec(20.0, 50.0), np.hypot(20.0, 50.0))

    def test_rejects_bad_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(path, az_deg=np.array([1.0, 2.0]))
            with self.assertRaises(ValueError):
                load_tiptilt_npz(path)


class TestBeamPointingParse(unittest.TestCase):
    def test_nominal_only(self):
        payload = {
            "timestamp": 1_700_000_100.0,
            "unit": "deg",
            "value": {"nominal": {"az_deg": 123.4, "el_deg": 45.6}},
        }
        record = parse_beam_pointing_payload(payload, beam=3)
        self.assertEqual(record.beam, 3)
        self.assertEqual(record.nominal, (123.4, 45.6))
        self.assertEqual(record.commanded_by_frequency_mhz, {})
        self.assertEqual(display_az_el(record, 55), (123.4, 45.6))

    def test_commanded_by_frequency_preferred(self):
        payload = {
            "timestamp": 1_700_000_100.0,
            "unit": "deg",
            "value": {
                "nominal": {"az_deg": 100.0, "el_deg": 40.0},
                "commanded_by_frequency_mhz": {
                    "55": {"az_deg": 101.5, "el_deg": 39.2},
                    "82": {"az_deg": 102.0, "el_deg": 38.0},
                },
            },
        }
        record = parse_beam_pointing_payload(json.dumps(payload), beam=1)
        self.assertEqual(display_az_el(record, 55), (101.5, 39.2))
        self.assertEqual(display_az_el(record, 82), (102.0, 38.0))
        # No matching commanded entry → nominal
        self.assertEqual(display_az_el(record, 70), (100.0, 40.0))
        self.assertEqual(set(record.commanded_by_frequency_mhz), {55, 82})

    def test_malformed_payloads(self):
        with self.assertRaises(ValueError):
            parse_beam_pointing_payload("not-json", beam=1)
        with self.assertRaises(ValueError):
            parse_beam_pointing_payload({"timestamp": 1.0, "value": {}}, beam=1)
        with self.assertRaises(ValueError):
            parse_beam_pointing_payload(
                {
                    "timestamp": 1.0,
                    "value": {
                        "nominal": {"az_deg": 1.0, "el_deg": 2.0},
                        "commanded_by_frequency_mhz": {"bad": {"az_deg": 1, "el_deg": 2}},
                    },
                },
                beam=1,
            )
        with self.assertRaises(ValueError):
            parse_beam_pointing_payload(
                {"timestamp": 1.0, "unit": "rad", "value": {"nominal": {"az_deg": 1, "el_deg": 2}}},
                beam=1,
            )

    def test_staleness(self):
        record = parse_beam_pointing_payload(
            {
                "timestamp": 1000.0,
                "value": {"nominal": {"az_deg": 1.0, "el_deg": 2.0}},
            },
            beam=2,
        )
        self.assertAlmostEqual(pointing_age_sec(record, now=1120.0), 120.0)
        self.assertFalse(is_pointing_stale(record, max_age_sec=120.0, now=1120.0))
        self.assertTrue(is_pointing_stale(record, max_age_sec=120.0, now=1120.1))
        # Non-finite limit (UI "all") never marks stale.
        self.assertFalse(
            is_pointing_stale(record, max_age_sec=float("inf"), now=1.0e12)
        )

    def test_read_beam_pointing_missing_and_present(self):
        client = FakeEtcd({})
        self.assertIsNone(read_beam_pointing(7, client=client))

        good = {
            "timestamp": time.time(),
            "value": {"nominal": {"az_deg": 10.0, "el_deg": 20.0}},
        }
        client = FakeEtcd({"/mon/beamformer/7/pointing": json.dumps(good).encode()})
        record = read_beam_pointing(7, client=client)
        self.assertIsNotNone(record)
        self.assertEqual(record.nominal, (10.0, 20.0))

        client = FakeEtcd({"/mon/beamformer/7/pointing": b"{"})
        self.assertIsNone(read_beam_pointing(7, client=client))


class TestActiveBeams(unittest.TestCase):
    def test_beam_number_from_recorder_id(self):
        self.assertEqual(beam_number_from_recorder_id("dr1"), 1)
        self.assertEqual(beam_number_from_recorder_id("dr12"), 12)
        self.assertIsNone(beam_number_from_recorder_id("drx"))
        self.assertIsNone(beam_number_from_recorder_id("drt1"))
        self.assertIsNone(beam_number_from_recorder_id(""))

    def test_list_active_beam_numbers(self):
        def fake_discover(*, type, age_limit):
            self.assertEqual(type, "power")
            self.assertEqual(age_limit, 90.0)
            return ["dr3", "dr1", "drx", "dr3", "other"]

        beams = list_active_beam_numbers(age_limit=90.0, discover_recorders=fake_discover)
        self.assertEqual(beams, [1, 3])

    def test_list_beam_numbers_from_pointing_keys(self):
        client = FakeEtcd(
            {
                "/mon/beamformer/1/pointing": b"{}",
                "/mon/beamformer/3/pointing": b"{}",
                "/mon/beamformer/2/status": b"{}",
                "/mon/beamformer/tiptilt": b"{}",
            }
        )
        self.assertEqual(
            list_beam_numbers_from_pointing_keys(client=client), [1, 3]
        )


if __name__ == "__main__":
    unittest.main()
