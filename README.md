# Radio Flux Solver

Code for calibrating/validating (low-frequency) radio flux data, based on reference surveys.

End-to-end; takes care of the finding, matching, and calibration of sources based on spectral index analysis.

#### Features
- Both catalog and image file support for flux validation
- Overview plots to showcase and filter for location/beam dependant effects
- Fixed powerlaw extrapolation in the basic mode
- Either two-point spectral index fitting or three-or-more-point index + curvature estimation advanced modes

#### Installation
You can install the required catalog files using
```bash
uv run install.py
```
or download them from source: `https://huggingface.co/datasets/EV4g/Radio-Flux-Solver-Catalogs/resolve/main/catalogs.tar.gz`.

#### Usage
You can validate a .fits file using:
```bash
uv run flux_calibrator_cli.py path/to/file.fits
```

For example:
```bash
uv run flux_calibrator_cli.py catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits --freq 144.6e6 --save-plots
```
will take the LoTSS DR3 data and reference it against all other available surveys, then save the inspection plots.
Since we are giving it a catalog, we also need to pass a frequency value (default unit is Hz, but can be specified with --freq-unit). For images, this can be inferred from the header.

A typical image analysis run would look something like:
```bash
uv run flux_calibrator_cli.py image.fits -c 4 --minimum-position-error 1 --minimum-frequency-spacing 1e6 --spatial-filter --save-plots
```
Which uses `combination-size=4` (matching of 3 catalogs to estimate the fourth) to estimate statistical curvature, sets a minimum positional error threshold of `1"`, a minimum frequency-spacing threshold of `1e6 Hz`, and pre-filters the catalogs to the image's coverage.

Currently supported arguments are:
```bash
[-h] [--scale SCALE] [--catalogs CATALOGS] [--anchor-name ANCHOR_NAME] [--freq FREQ] [--freq-unit {Hz,MHz,GHz}]
                              [--combination-size COMBINATION_SIZE] [--spectral_damping_factor SPECTRAL_DAMPING_FACTOR] [--nsigma NSIGMA] [--snr-lower-limit SNR_LOWER_LIMIT]
                              [--minimum-points MINIMUM_POINTS] [--minimum-position-error MINIMUM_POSITION_ERROR] [--spectral-index-theory SPECTRAL_INDEX_THEORY]
                              [--minimum-frequency-spacing MINIMUM_FREQUENCY_SPACING] [--reference-file REFERENCE_FILE] [--spatial-filter] [--no-reload-cache]
                              [--save-plots] [--debug] [--thres-arc THRES_ARC] [--n-jobs N_JOBS] [--logging] [--output-dir OUTPUT_DIR] [--seed SEED]
```

#### Currently included reference survey catalogs

| Catalog Name  | Frequency (MHz) | Notes   |
|---------------|-----------------|---------|
| vlssr         | 73.8            |         |
| lofar_dr3     | 144.6           |         |
| tgss          | 150             |         |
| gleam_x_gp    | 200             |         |
| gleam_300     | 300             |         |
| wenss         | 325             |         |
| vcss          | 340             |         |
| txs           | 365             |         |
| racs_gal      | 887.5           | Galactic part of racs_low |
| racs_low      | 887.5           |         |
| apertif       | 1355            |         |
| meerkat       | 1359.7          | SMGPS   |
| racs_mid      | 1367.5          |         |
| nvss          | 1400            |         |
| first         | 1400            |         |
| racs_high     | 1655.5          |         |
| vlass         | 3000            |         |
