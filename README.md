# Radio Flux Solver

Code for calibrating (low-frequency) Radio catalog-or-image flux, based on reference survey data.

End-to-end; takes care of the finding, matching, and calibration of sources based on spectral index analysis.

#### Features
- Both catalog and image file support for flux validation
- Overview plots to showcase and filter for location/beam dependant effects
- Fixed powerlaw extrapolation in the basic mode
- Either two-point spectral index fitting or three-or-more-point index + curvature estimation advanced modes

#### Currently nicluded reference survey catalogs

| Catalog Name  | Frequency (MHz) |
|---------------|-----------------|
| vlssr         | 73.8            |
| lofar_dr3     | 144.6           |
| tgss          | 150             |
| gleam_x_gp    | 200             |
| gleam_300     | 300             |
| wenss         | 325             |
| vcss          | 340             |
| txs           | 365             |
| racs_low      | 887.5           |
| apertif       | 1355            |
| meerkat       | 1359.7          |
| racs_mid      | 1367.5          |
| nvss          | 1400            |
| racs_high     | 1655.5          |
| vlass         | 3000            |
