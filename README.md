# Image-Plane Correction

This is a repository containing code/algorithms for image-plane correction, intended for use with the OVRO-LWA radio telescope.
This research is being performed by Zachary Huang as part of Caltech's 2024 SURF program (under the mentorship of Casey Law, Gregg Hallinan, and others).

# Libraries
We use [JAX](https://github.com/google/jax) to get easy GPU acceleration for matrix operations.
[Numpy](https://github.com/numpy/numpy) is still used in various places since some libraries don't support JAX arrays as a drop-in replacement.
We use [Astropy](https://github.com/astropy/astropy) for astronomical operations and [PyBDSF](https://github.com/lofar-astron/PyBDSF) for source detection.
[Matplotlib](https://github.com/matplotlib/matplotlib) is used for plotting.

[OpenCV](https://github.com/opencv/opencv) ([with extra modules](https://github.com/opencv/opencv_contrib)) is used for optical flow.
However, note that for CUDA support, OpenCV must be built from scratch.
This can be a bit of a pain, so I wrote down the installation steps that worked for me in `docs/opencv.md`.

# Development

All of the project dependencies (except for OpenCV) are specified in `pyproject.toml`.
To install these dependencies along with the source code, enter the project directory and run `pip install .`.
If you wish to run the notebooks in `notebooks/`, I would recommend using Jupyter Lab (which you can install along with the project dependencies with `pip install .[dev]`.
For the sake of development, I would also recommend adding the `-e` flag to the pip commands above so that any changes made to the source code in this repository are immediately reflected by the scripts/notebooks (instead of requiring a "reinstall" of the package).

For GPU acceleration, ensure that JAX is installed with [CUDA support](https://jax.readthedocs.io/en/latest/installation.html#installation).
As mentioned previously, OpenCV must be built from source for CUDA support and should be accessible as `cv2` for the optical flow code to work properly.

# Live tip-tilt warp maps (`watch_calcflow_warp`)

`scripts/watch_calcflow_warp.py` (and `scripts/mock_calcflow_warp_arrival.py`) poll
pipeline `*.fits.fz` snapshots, run `calcflow`, and write a **tip-tilt named-array
`.npz`** suitable for `mnc.tiptilt` (`az_deg`, `el_deg`, `dra_arcsec`,
`ddec_arcsec`, `obstime_unix`). By default the absolute path is published with
`mnc.tiptilt.set_tiptilt` under etcd `/mon/beamformer/tiptilt` (with a TTL; trackers
fall back to zero shift if the key expires while the file may still exist on disk).

```bash
PYTHONPATH=src python scripts/watch_calcflow_warp.py \
  --watch-dir /path/to/snapshots/55MHz \
  --output-dir /path/to/warps \
  --warp-filename warp_55MHz.npz \
  --cleaned --process-existing
```

Use `--no-publish-tiptilt` to write the NPZ without updating etcd. Frequency for
publish is taken from `_(\d+)MHz` in the warp filename (or `--frequency-mhz`).
See `scripts/watch_calcflow_warp.service` for a systemd `--user` unit template.

# Live dewarp map web app (`serve_dewarp_map`)

Panel + HoloViews app showing the tip-tilt vector field in **zenithal equal-area**
projection, with optional marks for active power beams. Requires viz extras
(`pip install '.[viz]'`) plus runtime access to the warp NPZ; live beam marks need
`mnc` / `etcd3` and beam pointing published by mnc
`publish_commanded_pointing` under `/mon/beamformer/{N}/pointing`.

```bash
# Install viz stack once (panel, holoviews, bokeh, param):
pip install -e '.[viz]'

# Serve against a local NPZ (path preferred over etcd for the UI):
PYTHONPATH=src python scripts/serve_dewarp_map.py \
  --warp-path /fast/claw/warps/warp_55MHz.npz \
  --frequency-mhz 55 \
  --poll-seconds 3
```

Then open `http://127.0.0.1:5007/` once in a browser. Pass `--show` only if you
want the script to open that URL for you — do not also click the printed URL
(that opens a second tab). Omit `--warp-path` to resolve the map via
`mnc.tiptilt.get_tiptilt_path(frequency_mhz)` when the tip-tilt catalog is
published. Beam discovery uses `mnc.ezdr.discover_recorders(type='power')`.

**Pointing display rule** (app frequency `F` MHz): use
`value.commanded_by_frequency_mhz["F"]` when present, else `value.nominal`.
Pointings older than `--max-pointing-age-sec` (default 120) are marked stale.
Offline UI work without etcd: omit `--demo-pointings` for map-only, or pass a
JSON file of `/mon/beamformer/{N}/pointing` payloads (`{"beams": {"N": {...}}}`)
when you need synthetic marks. The checked-in
`tests/fixtures/demo_pointings_55mhz.json` is empty by default.

See `scripts/serve_dewarp_map.service` for a systemd `--user` unit template.
**Dependencies to keep in mind:** tip-tilt etcd TTL (UI can still read `--warp-path`
after TTL expiry); beam marks require an mnc build that publishes
`/mon/beamformer/{N}/pointing` from `set_beam_pointing(s)`.

# Tuning optical-flow Brox parameters (alpha / gamma)

Dense optical flow uses OpenCV Brox parameters ``alpha`` (smoothness) and ``gamma``
(intensity gradient influence). Defaults in ``calcflow`` match historical usage; for a
given instrument / sky model you can grid-search or refine them with:

```bash
PYTHONPATH=src python scripts/optimize_alpha_gamma.py --search \
  --images path/to/image1.fits path/to/image2.fits --cleaned \
  --output-json tuning_results.json
```

This evaluates a coarse geometric grid (defaults roughly \(10^{-1}\)–\(10^{1}\) for ``alpha``,
\(10^{0}\)–\(10^{3}\) for ``gamma``, configurable via ``--alpha-min`` / ``--gamma-max`` / steps).
Each run calls ``calcflow`` per image and parameter pair; runtime scales with
``N_images × N_alpha × N_gamma`` (plus optional ``--refine`` L-BFGS-B iterations on
``log(alpha), log(gamma)`` within the grid bounds).

The JSON output (schema ``image_plane_correction.optimize_alpha_gamma.v4``) stores per-row
metrics, ``composite_objective`` (structure score plus QA weighting), and an aggregated
``recommended`` pair based on median composite across images (tie-break: QA pass rate,
then lower in-band curl/div ratio). Use ``--alphas`` / ``--gammas`` for a fixed small grid
instead of ``--search`` when you already know candidates.

Optional **catalog astrometry QA** compares **PyBDSF**-measured sources in the **raw** and **dewarped**
images to your reference catalog (same ``--catalog`` / ``--catalog-path`` as ``calcflow``).
Rows gain prefixed keys such as ``catalog_qa_raw_median_arcsec``,
``catalog_qa_dewarped_median_arcsec``, and ``catalog_qa_delta_median_arcsec`` when enabled:

```bash
PYTHONPATH=src python scripts/optimize_alpha_gamma.py --search \
  --images path/to/image.fits --psf path/to/psf.fits \
  --catalog-qa \
  --catalog VLSSR --catalog-path /path/to/catalog.txt \
  --output-json tuning_results.json
```

Beam major/minor FWHM in degrees are required for PyBDSF (from ``BMAJ`` / ``BMIN`` in the image FITS header or ``--catalog-qa-beam-deg BMAJ,BMIN``). Optional tuning flags include ``--catalog-qa-bdsf-thresh`` (``hard``, ``fdr``, or ``auto``), ``--catalog-qa-bdsf-thresh-isl``, ``--catalog-qa-bdsf-thresh-pix``, ``--catalog-qa-bdsf-minpix-isl``, and ``--catalog-qa-bdsf-ncores`` (default 4).

Shared implementations live in ``image_plane_correction.qa`` (catalog astrometry
metrics and bright-source logging helpers). The tuning script calls these automatically when
you pass ``--catalog-qa``. From Python you can run ``calcflow(..., catalog_qa=True)`` with
default ``catalog_qa_params=None`` (uses :class:`~image_plane_correction.qa.CatalogAstrometryQAParams`).
Metrics are stored on ``flow.catalog_qa_metrics``; optionally pass ``quality_metrics={}`` to merge the same keys into your dict as well.
