from __future__ import annotations

"""
Primary-beam correction helpers.

This module provides a stable API used by ``image_plane_correction.catalogs``:
- ``BEAM_PATH``
- ``BeamModel``
- ``_get_beam()`` singleton loader
"""

from .extractor_pb_75 import BEAM_PATH, BeamModel

_beam_model: BeamModel | None = None


def _get_beam() -> BeamModel:
    global _beam_model
    if _beam_model is None:
        _beam_model = BeamModel(BEAM_PATH)
    return _beam_model

