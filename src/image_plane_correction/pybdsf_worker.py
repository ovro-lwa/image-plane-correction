"""
Run PyBDSF in a clean interpreter (``python -m image_plane_correction.pybdsf_worker``).

Used when JAX is already loaded in the parent process: PyBDSF uses ``multiprocessing``
with fork-based workers, which is unsafe after JAX has started background threads.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def main() -> None:
    import numpy as np
    import bdsf

    ap = argparse.ArgumentParser(description="PyBDSF worker for image-plane-correction.")
    ap.add_argument("fits_path", help="Path to FITS image on disk.")
    ap.add_argument("out_npy", help="Path to write Nx2 float64 pixel centres (.npy).")
    ap.add_argument("kw_json", help="JSON object: PyBDSF kwargs plus __min_separation_px and __n_sources_max.")
    args = ap.parse_args()

    raw = json.loads(args.kw_json)
    min_sep = float(raw.pop("__min_separation_px", 0.0))
    nmax = raw.pop("__n_sources_max", None)
    kw = _deserialize_kw(raw)

    try:
        img = bdsf.process_image(args.fits_path, **kw)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    from image_plane_correction.source_detection import (
        _gaussian_rows_from_bdsf_image,
        _nms_xy_flux_sorted,
    )

    xy_all, flux_all = _gaussian_rows_from_bdsf_image(img)
    xy = _nms_xy_flux_sorted(
        xy_all,
        flux_all,
        min_separation_px=min_sep,
        max_keep=nmax,
    )
    np.save(args.out_npy, xy)


def _deserialize_kw(raw: dict[str, Any]) -> dict[str, Any]:
    kw = dict(raw)
    if "beam" in kw:
        kw["beam"] = tuple(float(x) for x in kw["beam"])
    return kw


if __name__ == "__main__":
    main()
