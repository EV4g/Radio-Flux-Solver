# Documentation

## Example usage

### Small image-input
Let's say we have an image `vla_cygnus.fits` observed at 336 MHz, and we want to see if the scale is correct relative to reference catalogs. An appropriate command would look like:
```bash
uv run flux_calibrator_cli.py vla_cygnus.fits --freq 336e6 --save-plots
```

If we have more information, and for example know the sources in this region have a spectral index closer to -0.7, we only want to compare it to a couple handpicked catalogs, and we know there will be few sources, the following command would be run:
```bash
uv run flux_calibrator_cli.py vla_cygnus.fits --freq 336e6 --catalogs wenss,vcss,nvss -c 2 --save-plots --save-csv
```

Let's say we, via a different method, figured out that the image likely has to be re-scaled by 1.4, a quick test to verify could be:
```bash
uv run flux_calibrator_cli.py vla_cygnus.fits --freq 336e6 --scale 1.4
```


### Compare the validity of LoTSS
Let's say we just made a catalog of the entire LoTSS DR3 survey, and want to compare whether or not its statistical flux level agrees with legacy surveys. Knowing that LoTSS DR3 is one of the main included catalogs, an appropriate command would be:
```bash
uv run flux_calibrator_cli.py catalogs/LoTSS_DR3_v1.0.srl_clean.fits --freq 144e6 --save-plots
```

If we want to see how LoTSS DR3 behaves for the galactic-plane, and assuming we have a .fits image of that region, an appropriate command would be:
```bash
uv run flux_calibrator_cli.py catalogs/LoTSS_DR3_v1.0.srl_clean.fits --freq 144e6 --catalogs vlssr,gleam_x_gp,racs_gal,meerkat --reference-file galactic_plane.fits --save-plots
```

If we want to measure curvature as function of ra,dec for the whole available sky, and want more signal-to-noise by being more lenient in the source-matcher, we could run:
```bash
uv run flux_calibrator_cli.py catalogs/LoTSS_DR3_v1.0.srl_clean.fits --freq 144e6 -c 4 --n-sigma 4 --save-plots
```

## Arguments
To run the program, run `uv run flux_calibrator_cli.py` followed by a selection of arguments.

### Main arguments
**`catalog`**\
This is the main argument to pass: `uv run flux_calibrator_cli.py catalog.fits`, and represents the data you want to compare all reference catalogs. Can be an actual .fits catalog, or a .fits image. In the latter case, PyBDSF will be run to turn it in to a catalog.

**`-h`, `--help`**\
This will print out all available arguments.

**`--catalogs`**\
This represents the catalogs to which the user's data will be compared.
Can be either a preset (all, default, small), or a comma-separated list of catalog names, like: `--catalogs wenss,nvss,vlass`.

**`-f, --freq`**\
The central frequency of the data. If the data is a .fits image, this can likely be infered from the header without extra input. If you give it a catalog though, you will need to specify it. Unless specified with `--freq-unit`, the unit will be assumed to be in Hz. Example: `--freq 144e6`.

**`--freq-unit`, default: Hz**\
The unit of the `--freq` input. Can be Hz, MHz, or GHz. Only relevant when also passing `--freq`.

**`-c`, `--combination-size`, default: 3**\
An important argument that specifies how many points are used in fitting.\
If `c=2`, then the code looks for sources that have atleast two data points for flux, at atleast two different frequencies. The correction factor is then based on a direct extrapolation of the catalog source to the input, using `spectral-index-theory`. Since it requires only two data points per source, this is stable for datasets with few sources, or when one knows the spectral-index precisely. It only outputs the correction factor, since all other parameters are fixed.\
If `c=3`, the code looks for sources with three data points. Two of those are used to fit for the spectral index, and the flux is then extrapolated to the frequency of the input source. Since the spectral index is fitted for, it can be used for further analysis.\
If `c=4`, the code looks for four data points per match. This allows it to fit for spectral-index, as well as curvature. It does require four points per match, meaning significantly fewer matches in total.

**`--nsigma`, default: 2**\
The code used error-based source matching. This parameter adjusts to which level sources are still considered 'matched'. A higher value means that sources that are further appart can still be matched.
Example: `--nsigma 3` means that if two sources are within 3 sigma of each other, based on their errors, they are considered the same source.

**`--snr-lower-limit`, default: 7**\
Only sources with a flux signal-to-noise above this limit are considered. Sources below this level are ignored.

**`--minimum-frequency-spacing`, default: 10e6**\
To avoid weird behaviour with catalogs that are extremely close to eachother in frequency space, there exists a lower limit. Two catalogs which are spaced out closer than this limit will not be matched. If a catalog is closer than the limit to the input data, it will fully be ignored. The default value is 10e6 Hz.

**`--scale`, default: 1**\
The flux scale of the input data. Can be used if you know beforehand that a scaling should be applied to the data.


### Fitting arguments
**`--spectral-model`, default: CPL**\
This argument is for changing the spectral model that is used during fitting of the spectra. Options are: CPL, FFA, SSA. CPL is a simple curved-powerlaw, which in most cases will suffice, and is therefore the default.
FFA is a free-free turn-over model, and SSA is a more complex model with two powerlaws. CPL and FFA's are fast, while SSA currently requires `curve-fit`, and is therefore incredibly slow until further updates.

**`--spectral-index-theory`, default: -0.8**\
The theoretical value for spectral index. If `--combination-size=2`, this will be used for the extrapolation. In other cases, it will be used to reject outliers that are significantly deviant from this theoretical value.

**`--spectral-curvature-theory`, default: 0**\
The theoretical value for spectral curvature. Defaults to 0, but can be set to influence the lower-order fits if a theory value is known for the region. Will be fitted for if `--combination-size>=4`

**`--tau-freefree-theory`, default: 0**\
Theoretical value for the tau_ff parameter when using `--spectral-model=FFA`.

**`--pivot-freq-theory`, default: 100e6**\
Theoretical value for the reference-frequency used for fitting. Determines the location of the curvature in `--spectral-model=CPL`, as well as the location of the turnover when using `--spectral-model=FFA`. Defaults to 100 MHz, which is a sensible value for the default powerlaw model.

**`--spectral-index-thin-theory`, default: -0.5**\
Theoretical value for the spectral-index-thin in the case `--spectral-model=SSA`.

**`--spectral-index-thick-theory`, default: 2.5**\
Theoretical value for the spectral-index-thick in the case `--spectral-model=SSA`.


### Logging
**`--save-plots`**\
If this argument is given, some inspection plots will be saved locally. These can be used to analyse the spatial behaviour of the correction factor, as well as get an idea of the uncertainty.

**`--save-csv`**\
If this argument is given, all matched sources will be saved into a csv file, with coordinates and other relevant values.

**`--logging`**\
Write logs to file, instead of displaying in a terminal. Can be used to 

**`--output-dir`, default: ./logs/**\
Option to change the output directory. Used when any of `--logging`, `--debug` or `--save-plots` is specified.


### Advanced arguments
**`--fitting-order`, default: combination-size - 1**\
Order of the fit used to extrapolate to the input frequency. Default behaviour is pinned to `--combination-size`, but can be manually set lower while keeping a high combination size. Resulting extrapolation will then be an average of lower-order fits.

**`--spectral_damping_factor`, default: 2**\
Value used to downweight spectral indices very far away from the theoretical value. A higher damping factor results in more sources being downweighted, while a value of 0 disables downweighting entirely. A value > 0 is recommended to get rid of source mis-matches as well as unphysical values. Can be disabled to specifically look for steep spectrum sources.

**`--minimum-points`, default: 3**\
Ignore catalog matchings if the total number of sources in that catalog match is below this threshold. This is separate from the *total* number of matches from all catalogs, and separate from the total number of matches from this catalogs. Example [wenss, nvss, vlass] --> 2 sources --> ignore this match; [wenss, nvss, racs_high] --> 500 sources --> keep.

**`--minimum-position-error`, default: None**\
Normally the source location error is taken from the error values directly in the catalogs. This argument overrides the lower bound of these values. This can be useful if the catalog reports relative errors, but you know that there might be an astrometric offset that is larger than those errors. Values given are in arcseconds.

**`--reference-file`, default: None**\
If a reference .fits image is provided, the output is generated for within this region. Useful if you have a catalog and are only interested in a certain sub-region.

**`--spatial-filter`, default: False**\
if an .fits image is given, this setting will pre-cut all main catalogs to the extent of the image, throwing away sources outside of it. This will speed up the matching, but defaults to False until further testing.

**`--thres-arc`, default: None**\
If this value is set, the error-based source matching is ignored, and instead it goes to a linear nearest-neighbor matching system. Input values are in arcseconds.

**`--crowd-radius-arc`, default: None**\
This value can be set to downweight, or ignore, sources in busy areas. Sources withing this (arcseconds) radius receive a crowding penalty in the weighting of the correction factor.

**`--n-jobs`, default: -1**\
Number of cpu cores to use, defaults to all available cores.

**`--debug`, default: False**\
If this argument is passed, debug plots will be saved per set of matched catalogs, to identify per-catalog effects. Note, this will significantly slow down the code.

**`--no-reload-cache`, default: False**\
When inputting an image, the pybdsf catalog is saved locally as cache, to speed-up re-runs. If this argument is given, the cache is ignored, and everything is re-run from scratch.

**`--seed`, default: random**\
Seed for one of the plots showcasing a random asortment of spectra.

**`--anchor-name`, default: None**\
Option to change the name of the input file in the debug plots. Can be used for clarity.
