"""
Compatibility shim for historical imports.

Prefer importing from ``image_plane_correction.pb_correct``.
"""

from image_plane_correction.pb_correct import BEAM_PATH, BeamModel, _get_beam

__all__ = ["BEAM_PATH", "BeamModel", "_get_beam"]

