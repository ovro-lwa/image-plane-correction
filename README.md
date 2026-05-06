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

The JSON output (schema ``image_plane_correction.optimize_alpha_gamma.v2``) stores per-row
metrics, ``composite_objective`` (structure score plus QA weighting), and an aggregated
``recommended`` pair based on median composite across images (tie-break: QA pass rate,
then lower in-band curl/div ratio). Use ``--alphas`` / ``--gammas`` for a fixed small grid
instead of ``--search`` when you already know candidates.
