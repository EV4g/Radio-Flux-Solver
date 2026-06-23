# Radio Flux Solver

Code for calibrating/validating (low-frequency) radio flux data, based on reference surveys.

End-to-end; takes care of the finding, matching, and calibration of sources based on spectral index analysis.

#### Features
- Both catalog and image file support for flux validation
- Overview plots to showcase and filter for location/beam dependant effects
- Fixed powerlaw extrapolation in the basic mode
- Either two-point spectral index fitting or three-or-more-point index + curvature estimation advanced modes

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
| racs_high     | 1655.5          |         |
| vlass         | 3000            |         |

#### Usage
You can validate a .fits file using:
```
uv run flux_calibrator_cli.py path/to/file.fits
```
Currently supported arguments are:
```
[-h] [--scale SCALE] [--catalogs CATALOGS] [--anchor-name ANCHOR_NAME] [--freq FREQ] [--freq-unit {Hz,MHz,GHz}]
                              [--combination-size COMBINATION_SIZE] [--spectral_damping_factor SPECTRAL_DAMPING_FACTOR] [--nsigma NSIGMA] [--snr-lower-limit SNR_LOWER_LIMIT]
                              [--minimum-points MINIMUM_POINTS] [--spectral-index-theory SPECTRAL_INDEX_THEORY] [--minimum-frequency-spacing MINIMUM_FREQUENCY_SPACING]
                              [--reference-file REFERENCE_FILE] [--spatial-filter] [--no-reload-cache] [--save-plots] [--debug] [--thres-arc THRES_ARC] [--n-jobs N_JOBS]
                              [--logging] [--output-dir OUTPUT_DIR] [--seed SEED]
```
