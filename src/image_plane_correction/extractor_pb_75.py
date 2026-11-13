"""
Minimal primary-beam model interface used by this repository.

The main pipeline only requires:
- ``BEAM_PATH`` constant (string path to an HDF5 beam model)
- ``BeamModel`` with ``get_response(ra_deg, dec_deg, obs_time, freq_hz)``

This is intentionally a lightweight, import-safe subset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

# Default OVRO location (used to convert RA/Dec -> AltAz)
OVRO_LOC = EarthLocation(lat=37.23977727 * u.deg, lon=-118.2816667 * u.deg, height=1222 * u.m)

# Default beam model path (can be overridden by setting environment variable)
BEAM_PATH = os.environ.get(
    "OVRO_LWA_BEAM_H5",
    "/lustre/gh/calibration/pipeline/reference/beams/OVRO-LWA_MROsoil_updatedheight.h5",
)


@dataclass
class _BeamGrid:
    freq_hz: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    stokes_i_norm: np.ndarray


class BeamModel:
    """
    Interpolated Stokes-I primary-beam model backed by an HDF5 grid.

    The HDF5 file is expected to contain:
    - ``freq_Hz`` (1D)
    - ``theta_pts`` (1D), in degrees or radians
    - ``phi_pts`` (1D), in degrees or radians
    - ``X_pol_Efields/etheta``, ``X_pol_Efields/ephi``, ``Y_pol_Efields/etheta``, ``Y_pol_Efields/ephi``
      (complex arrays) from which Stokes I is computed.
    """

    def __init__(self, h5_path: str):
        self.path = str(h5_path)
        self._grid: _BeamGrid | None = None
        self._interp: Any | None = None

    def _load(self) -> None:
        if self._interp is not None:
            return

        import h5py
        from scipy.interpolate import RegularGridInterpolator

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Beam file not found: {self.path}")

        with h5py.File(self.path, "r") as hf:
            fq_orig = np.asarray(hf["freq_Hz"][:], dtype=float)
            th_orig = np.asarray(hf["theta_pts"][:], dtype=float)
            ph_orig = np.asarray(hf["phi_pts"][:], dtype=float)
            Exth = hf["X_pol_Efields/etheta"][:]
            Exph = hf["X_pol_Efields/ephi"][:]
            Eyth = hf["Y_pol_Efields/etheta"][:]
            Eyph = hf["Y_pol_Efields/ephi"][:]

        # Convert to degrees if stored in radians.
        if np.nanmax(th_orig) < 10.0:
            th_orig = np.degrees(th_orig)
        if np.nanmax(ph_orig) < 10.0:
            ph_orig = np.degrees(ph_orig)

        stokes_i = (np.abs(Exth) ** 2 + np.abs(Exph) ** 2 + np.abs(Eyth) ** 2 + np.abs(Eyph) ** 2).astype(float)

        fq_idx = np.argsort(fq_orig)
        th_idx = np.argsort(th_orig)
        ph_idx = np.argsort(ph_orig)

        stokes_i_s = stokes_i[fq_idx, :, :][:, th_idx, :][:, :, ph_idx]
        zenith_idx = int(np.argmin(np.abs(th_orig[th_idx])))
        zen = stokes_i_s[:, zenith_idx, 0].astype(float)
        zen = np.where(zen == 0, 1.0, zen)
        norm = stokes_i_s / zen[:, np.newaxis, np.newaxis]

        self._grid = _BeamGrid(
            freq_hz=fq_orig[fq_idx],
            theta_deg=th_orig[th_idx],
            phi_deg=ph_orig[ph_idx],
            stokes_i_norm=norm,
        )
        self._interp = RegularGridInterpolator(
            (self._grid.freq_hz, self._grid.theta_deg, self._grid.phi_deg),
            self._grid.stokes_i_norm,
            bounds_error=False,
            fill_value=0.0,
        )

    def get_response(self, ra: Any, dec: Any, obs_time: Any, freq_hz: float) -> np.ndarray:
        self._load()
        assert self._interp is not None

        t = Time(obs_time, location=OVRO_LOC)
        sc = SkyCoord(ra, dec, unit="deg")
        altaz = sc.transform_to(AltAz(obstime=t, location=OVRO_LOC))

        az = np.asarray(altaz.az.deg, dtype=float)
        el = np.asarray(altaz.alt.deg, dtype=float)
        theta = 90.0 - el
        phi = np.mod(az, 360.0)

        freq = float(freq_hz)
        if np.isscalar(theta):
            pts = np.array([freq, float(theta), float(phi)], dtype=float)
        else:
            pts = np.column_stack((np.full(theta.size, freq, dtype=float), theta.ravel(), phi.ravel()))

        resp = np.asarray(self._interp(pts), dtype=float).reshape(np.shape(theta))
        resp = np.where(el < 10.0, np.nan, resp)
        return resp

