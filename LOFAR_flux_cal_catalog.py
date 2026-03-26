from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import copy
from functions import match_catalogs_2D, compute_fluxcal_statistics, get_spectral_index, calculate_contour_statistics, get_combinations, get_pos_err_deg
from astropy.table import Table
from scipy.stats import chi2
from itertools import combinations

# wrapper class for incoming Table data
class catalog:
    def __init__(self, catalog, freq_hz=None, name=None, store_raw=False):
        # if a string is given, try to read it as Path
        if isinstance(catalog, str):
            try:
                catalog = Table.read(os.getcwd()+catalog)
            except Exception as e:
                raise FileNotFoundError(f"Could not read catalog: {e}")

        self.flux       = np.array(catalog['flux_jy'])
        self.e_flux     = np.array(catalog['e_flux_jy'])
        self.flux_unit  = str(catalog['flux_jy'].unit)

        self.ra         = np.array(catalog['ra'])
        self.dec        = np.array(catalog['dec'])

        try:
            self.e_ra   = np.array(catalog['e_ra'])
            self.e_dec  = np.array(catalog['e_dec'])
            self.e_ra[np.where(np.isnan(self.e_ra))] = 0   # sanitize NaNs
            self.e_dec[np.where(np.isnan(self.e_dec))] = 0 # sanitize NaNs
        except:
            self.e_ra   = None
            self.e_dec  = None

        self.freq       = freq_hz
        self.freq_unit  = 'Hz'
        self.name       = name

        self.raw = catalog if store_raw else None

    def create_subset(self, valid):
        subset = copy.deepcopy(self)
        subset.flux   = self.flux[valid]
        subset.e_flux = self.e_flux[valid]
        subset.ra     = self.ra[valid]
        subset.dec    = self.dec[valid]
        if self.e_ra is not None:  subset.e_ra  = self.e_ra[valid]
        if self.e_dec is not None: subset.e_dec = self.e_dec[valid]
        return subset

# wrapper class for passable parameters
class config:
    def __init__(self, thres_arc, spectral_damping_factor, snr_lower_limit, spectral_index_theory=-0.7, minimum_points=2, nsigma=3):
        self.thres_arc = thres_arc
        self.spectral_damping_factor = spectral_damping_factor
        self.snr_lower_limit = snr_lower_limit
        self.minimum_points = minimum_points
        self.spectral_index_theory = spectral_index_theory
        self.nsigma = nsigma

"""Given arrays of RA/Dec (degrees) and a FITS file, return a boolean array of which sources fall within the image footprint."""
def sources_in_fits(ra_deg, dec_deg, fn):
    with fits.open(fn) as hdul:
        w  = WCS(hdul[0].header).celestial
        nx = hdul[0].header["NAXIS1"]
        ny = hdul[0].header["NAXIS2"]

    coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit=u.deg, frame="icrs")
    x, y   = w.world_to_pixel(coords)

    return (x >= 0) & (x < nx) & (y >= 0) & (y < ny)

"""Get two catalogs and return flux and snr (flux / e_flux)"""
def get_catalog_matched_flux(cat1, cat2, thres_arc=2):
    idx_cat1, idx_cat2 = match_catalogs_2D([cat1, cat2], thres_arc=thres_arc)

    flux1 = cat1.flux[idx_cat1]
    snr1 = flux1 / cat1.e_flux[idx_cat1]

    flux2 = cat2.flux[idx_cat2]
    snr2 = flux2 / cat2.e_flux[idx_cat2]

    return flux1, flux2, snr1, snr2

"""Load two catalogs and plot their relative fluxes, according to a powerlaw"""
def quick_compare_catalog(cat1, cat2, config):
    flux1, flux2, _, _ = get_catalog_matched_flux(cat1, cat2, thres_arc=config.thres_arc)
    spectral_flux_ratio, spectral_index_actual, x, log_ratio, scale_factor = compute_fluxcal_statistics(cat1.freq, cat2.freq, flux1, flux2, config.spectral_index_theory)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(flux2, flux1, s=10, alpha=0.7)
    ax[0].plot(x, x * spectral_index_actual, color='purple', ls='--', label="Data fit")
    ax[0].plot(x, x, color='black', ls='--', label="x = y")
    ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={config.spectral_index_theory})')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(f"{cat1.name} flux [Jy]"); ax[0].set_ylabel(f"{cat2.name} flux [Jy]")
    ax[0].legend()

    # Log-ratio histogram
    ax[1].hist(log_ratio, bins=30, edgecolor='k')
    ax[1].axvline(scale_factor, color='red', ls='--', label=f'Median = {scale_factor:.3f}')
    ax[1].set_xlabel(f"log10(S_{cat1.name} / S_{cat2.name})")
    ax[1].set_ylabel("N")
    ax[1].legend()
    plt.tight_layout()
    #plt.savefig("flux_scale_comparison.png", dpi=150)
    plt.show()

    print(f"Compared {cat1.name} to {cat2.name} \n")

"""compute the flux correction factor based on three given catalogs. Catalogs are matches, and the last two are used to calculate the spectral index
which is used to extrapolate what the first cat -should- be. The different between -should- and -is-, is the correction factor."""
def compute_flux_correction_factor(cats, config, anchor_catalog=None, debug=False):
    indices, quality = match_catalogs_2D(cats, thres_arc=config.thres_arc, return_quality=True, nsigma=config.nsigma)

    # if there are too few sources, return None
    if len(indices[0]) <= config.minimum_points:
        if debug: print(f"Error: no source-matches found between [{', '.join(f'{cat.name}' for cat in cats)}]")
        return None

    # if crowding parameter is available, remove overcrowded points; else skip
    if quality['n_crowd']:
        crowd_ok = np.ones(len(indices[0]), dtype=bool)
        for cat_idx, match_idx in enumerate(indices):
            nc = quality['n_crowd'].get(cat_idx)
            if nc is not None and len(nc): crowd_ok &= (nc[match_idx] == 0)
        indices = [index[crowd_ok] for index in indices]

    # create subsets of all catalogs, such that we can ignore (i0,i1,...) afterwards
    for index, cat in enumerate(cats): cats[index] = cat.create_subset(indices[index])

    # all pairwise separations, then take per-source maximum
    coords = [SkyCoord(ra=cat.ra * u.deg, dec=cat.dec * u.deg) for cat in cats]
    sig    = [get_pos_err_deg(cat) * 3600 for cat in cats]   # arcsec per-source

    pair_seps = {}
    p_weight  = np.ones(len(cats[0].ra))
    for a, b in combinations(range(len(cats)), 2):
        sep = coords[a].separation(coords[b]).arcsec
        pair_seps[(a, b)] = sep
        p_weight *= 1.0 - chi2.cdf((sep / np.hypot(sig[a], sig[b])) ** 2, df=2)

    max_sep = np.maximum.reduce(list(pair_seps.values()))

    # figure out for which catalog the correction factor should be calculated
    # if cat is given, match that, otherwise assume the first index
    other_index = [i for i in range(len(cats))]
    if anchor_catalog is None:
        anchor_index = 0
    else:
        try:
            anchor_index = cats.index(anchor_catalog)
        except:
            print("Anchor catalog missing from inputs")
            return None
    other_index.remove(anchor_index)

    # flux and flux error
    uncorrected_flux = cats[anchor_index].flux
    uncorrected_flux_error = cats[anchor_index].e_flux

    # calculate spectral indices based on available data
    # w.i.p. for case n=4, where curvature can be estimated
    match len(cats):
        case 2:
            spectral_indices = np.ones_like(cats[0].flux) * config.spectral_index_theory
        case 3:
            spectral_indices = get_spectral_index(cats[other_index[0]].flux, cats[other_index[1]].flux, cats[other_index[0]].freq, cats[other_index[1]].freq)
        case _:
            print("Not yet implemented")
            return None

    # fit is based on measuring between two points, linear is average between assuming theoretical extragalactic value for both
    extrapolated_flux_fit = get_flux_from_index(spectral_indices, cats[other_index[0]].flux, cats[anchor_index].freq, cats[other_index[0]].freq)
    
    # calculate the linear theoretical flux at anchor_index based on all the other catalogs, and average the result
    extrapolated_flux_linear = np.mean([get_flux_from_index(config.spectral_index_theory, cats[index].flux, cats[anchor_index].freq, cats[index].freq) for index in other_index], axis=0)

    if debug:
        for flux in np.array([cat.flux for cat in cats]).T:
            plt.plot(np.array([cat.freq for cat in cats])*1e-6, flux)

        plt.yscale('log')
        plt.ylabel(r"log$_{10}$(Jy)")
        plt.xlabel("Frequency (MHz)")
        plt.show()
        
    # correction factor is the factor to multiply the anchor_catalog flux by to get what it should be, based on the other catalogs
    correction_factor = extrapolated_flux_fit / uncorrected_flux
    snr = uncorrected_flux / uncorrected_flux_error
    
    # apply weighting factor per catalog, downweighting uncertain ones. To be depracated in favour
    # of a proper flux extrapolator and proper catalog uncertainty values.
    catalog_weight_factor = np.ones_like(snr)
    for cat in cats:
        if cat.name.lower() == "tgss": catalog_weight_factor *= 0.5
        if cat.name.lower() == "gleam_300": catalog_weight_factor *= 0.7
        if cat.name.lower() == "gleam_xgp": catalog_weight_factor *= 0.8
    
    # calculate average position over all catalogs instead of using the first catalog
    ra = np.average([cat.ra for cat in cats], axis=0)
    dec = np.average([cat.dec for cat in cats], axis=0)
    
    if debug:
        # compare -0.7 assumption versus fitted spectral indices
        mn, mx = min(np.min(extrapolated_flux_fit), np.min(extrapolated_flux_linear)), max(np.max(extrapolated_flux_fit), np.max(extrapolated_flux_linear))
        plt.scatter(extrapolated_flux_linear, extrapolated_flux_fit, c=spectral_indices)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(mn, mx)
        plt.ylim(mn, mx)
        plt.gca().set_box_aspect(1)
        plt.plot((mn, mx), (mn, mx), c='k', ls='--')
        plt.colorbar(label = r"Spectral index $\alpha$")
        plt.xlabel(f"{cats[0].name} linear flux (Jy)")
        plt.ylabel(f"{cats[0].name} fitted flux (Jy)")
        plt.title(f"{cats[0].name} "+r"flux, $\alpha$=-0.7 vs fitted")
        plt.show()

        # compare fitted spectral index with correction factor
        plt.scatter(spectral_indices, correction_factor, c=uncorrected_flux, norm='log')
        plt.yscale('log')
        plt.axvline(-0.7, ls='--', c='k')
        plt.axhline(1, ls='--', c='k')
        plt.colorbar(label='Flux (Jy)')
        plt.ylabel("Flux correction factor")
        plt.xlabel(r"Spectral index $\alpha$")
        plt.title("Flux correction as function of spectral index")
        plt.show()

    print(f"Completed set [{', '.join(f'{cat.name:9}' for cat in cats)}]", round(catalog_weight_factor[0], 2) if debug else "")

    return (spectral_indices, snr, correction_factor, extrapolated_flux_fit, catalog_weight_factor, max_sep, p_weight, ra, dec)

"""Calculate weighted correction factor based on per-point spectral indices, signal-to-noise, and correction factor"""
def calculate_weighted_correction_factor(spx, snr, catw, max_sep, config):
    # downweight sources with spectral indices far away from -0.7
    spectral_difference_factor = np.exp(-config.spectral_damping_factor * (spx - config.spectral_index_theory)**2)

    # discard low snr sources
    signal_to_noise_factor = snr
    signal_to_noise_factor[signal_to_noise_factor < config.snr_lower_limit] = 0

    # weighting based on separation between points
    separation_weight = np.exp(-(max_sep / config.thres_arc) ** 2)

    return spectral_difference_factor * signal_to_noise_factor * catw * separation_weight

"""Extrapolate flux based on two frequencies, one flux, and a spectral index"""
def get_flux_from_index(spectral_index, reference_flux, current_frequency, reference_frequency):
    return reference_flux * (current_frequency / reference_frequency) ** spectral_index

# lofar fits file
lofar_files = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0]


# load catalogs, names in order of acquisition, lofar last
racs_full      = catalog("/catalogs/racs/racs_clean.fits",                 887.5e6,    "racs")
meerkat_full   = catalog("/catalogs/meerkat/meerkat_clean.fits",           1359.7e6,   "meerkat")
vlssr_full     = catalog("/catalogs/vlssr/vlssr_clean.fits",               73.8e6,     "vlssr")
tgss_full      = catalog("/catalogs/tgss/tgss_clean.fits",                 150e6,      "tgss")
gleam_300_full = catalog("/catalogs/gleam_300/gleam_300_clean.fits",       300e6,      "gleam_300")
gleam_xgp_full = catalog("/catalogs/gleam_x_gp/gleam_x_gp_clean.fits",     200e6,      "gleam_xgp")
lofar_full     = catalog("/catalogs/lofar/lofar_sources_pipeline.fits",    144.6e6,    "lofar")
#lofar_full    = catalog("/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",   144.6e6,    "lofar")

# check for sources in current file
racs_valid      = sources_in_fits(racs_full.ra,      racs_full.dec,       lofar_files)
meerkat_valid   = sources_in_fits(meerkat_full.ra,   meerkat_full.dec,    lofar_files)
vlssr_valid     = sources_in_fits(vlssr_full.ra,     vlssr_full.dec,      lofar_files)
tgss_valid      = sources_in_fits(tgss_full.ra,      tgss_full.dec,       lofar_files)
gleam_300_valid = sources_in_fits(gleam_300_full.ra, gleam_300_full.dec,  lofar_files)
gleam_xgp_valid = sources_in_fits(gleam_xgp_full.ra, gleam_xgp_full.dec,  lofar_files)
lofar_valid     = sources_in_fits(lofar_full.ra,     lofar_full.dec,      lofar_files)

# remove all non-valid points to reduce syntax clutter
racs      = racs_full.create_subset(racs_valid)
meerkat   = meerkat_full.create_subset(meerkat_valid)
vlssr     = vlssr_full.create_subset(vlssr_valid)
tgss      = tgss_full.create_subset(tgss_valid)
gleam_300 = gleam_300_full.create_subset(gleam_300_valid)
gleam_xgp = gleam_xgp_full.create_subset(gleam_xgp_valid)
lofar     = lofar_full.create_subset(lofar_valid)

catalogs      = [racs, meerkat, vlssr, tgss, gleam_300, gleam_xgp, lofar]
catalogs_full = [racs_full, meerkat_full, vlssr_full, tgss_full, gleam_300_full, gleam_xgp_full, lofar_full]

#### Parameters
debug = True
default_config = config(thres_arc=2, spectral_damping_factor=5, snr_lower_limit=7, nsigma=2.5)
vlssr_config   = config(thres_arc=10, spectral_damping_factor=5, snr_lower_limit=7)

if debug:
    # full catalog plot
    for cat in catalogs_full:
        plt.hist(np.log10(cat.flux), alpha=0.6, bins=75, label=cat.name)
    plt.xlabel("log10(flux/Jy)")
    plt.ylabel("count")
    plt.yscale('log')
    plt.legend()
    plt.show()

    # cutdown catalog plot
    for cat in catalogs:
        plt.hist(np.log10(cat.flux), alpha=0.6, bins=25, label=cat.name)
    plt.xlabel("log10(flux/Jy)")
    plt.ylabel("count")
    plt.yscale('log')
    plt.legend()
    plt.show()

    # catalog as function of position
    for cat in catalogs:
        if len(cat.ra) > 0: plt.scatter(cat.ra, cat.dec, s=1, label=cat.name)
    plt.gca().set_box_aspect(1)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend()
    plt.show()

    """
    #### analysis
    quick_compare_catalog(lofar, meerkat, default_config)
    quick_compare_catalog(lofar, racs,    default_config)
    quick_compare_catalog(lofar, tgss,    default_config)
    quick_compare_catalog(lofar, vlssr,   vlssr_config)
    quick_compare_catalog(racs,  meerkat, default_config)
    """

#### variables
ras, decs = [], []              # positional coordinates
correction_factor_global = []   # ratio between read-out catalog0 flux and computed
spectral_index_global = []      # per-source two-point spectral index
fitted_flux = []                # catalog0 flux based on two-point spectral index extrapolation
signal_to_noise = []            # signal-to-noise (flux_jy / e_flux_jy)
catalog_weight_factor = []      # weighting factor based on systematic catalog uncertainty; to be deprecated
max_separation = []             # maximum per-source separation between all three matched catalog positions
point_probability = []          # probability of points matching

###################################################
#### catalog three-way combination auto-looper ####
###################################################
for combination in get_combinations(catalogs, size=3, required_index=6, skip_index=2):
    local_cats = [catalogs[i] for i in combination]

    output = compute_flux_correction_factor(local_cats, default_config, debug=debug)

    if output is not None:
        spx, snr, cor, flux, catw, max_sep, p_weight, ra, dec = output

        ras += [ra]; decs += [dec]
        correction_factor_global += [cor]
        spectral_index_global += [spx]
        fitted_flux += [flux]
        signal_to_noise += [snr]
        catalog_weight_factor += [catw]
        max_separation += [max_sep]
        point_probability += [p_weight]

###########################################################
#### plotting correction factor for each catalog combo ####
###########################################################
for spx, snr, cor, catw, max_sep in zip(spectral_index_global, signal_to_noise, correction_factor_global, catalog_weight_factor, max_separation):
    total_weighting_factor = calculate_weighted_correction_factor(spx, snr, catw, max_sep, default_config)
    Xi, Yi, Zi, px, py = calculate_contour_statistics(spx, cor, total_weighting_factor, logy=True)

    o = np.argsort(total_weighting_factor)

    fig, ax = plt.subplots()
    sc = ax.scatter(spx[o], cor[o], c=total_weighting_factor[o])
    ax.contour(Xi, Yi, Zi, levels=6, colors='red', alpha=0.7, linewidths=0.8)
    plt.colorbar(sc, label="Combined weighting factor")
    plt.yscale('log')

    plt.axhline(py, ls="--", color="gray")
    plt.axvline(px, ls="--", color="gray")

    plt.xlim(np.percentile(spx, 1), np.percentile(spx, 99))
    plt.ylim(np.percentile(cor, 1), np.percentile(cor, 99))
    plt.ylabel("Correction factor")
    plt.xlabel(r"Fitted spectral index $\alpha$")
    plt.title("Correction factor as function of fitted spectral index")
    plt.show()

    print(f"Spectral index: {round(px,3)}, correction factor: {round(py,3)}")

ras = np.concatenate(ras)
decs = np.concatenate(decs)
correction_factor_global = np.concatenate(correction_factor_global)
spectral_index_global = np.concatenate(spectral_index_global)
fitted_flux = np.concatenate(fitted_flux)
signal_to_noise = np.concatenate(signal_to_noise)
catalog_weight_factor = np.concatenate(catalog_weight_factor)
max_separation = np.concatenate(max_separation)


############################################################################
#### plotting correction factor based on all previous catalog matchings ####
############################################################################
total_weighting_factor = calculate_weighted_correction_factor(spectral_index_global, signal_to_noise, catalog_weight_factor, max_separation, default_config)
Xi, Yi, Zi, px, py = calculate_contour_statistics(spectral_index_global, correction_factor_global, total_weighting_factor, logy=True)

o = np.argsort(total_weighting_factor)

fig, ax = plt.subplots()
sc = ax.scatter(spectral_index_global[o], correction_factor_global[o], c=total_weighting_factor[o])
ax.contour(Xi, Yi, Zi, levels=6, colors='red', alpha=0.7, linewidths=0.8)
plt.colorbar(sc, label="Combined weighting factor")
plt.yscale('log')

plt.axhline(py, ls="--", color="gray")
plt.axvline(px, ls="--", color="gray")

plt.xlim(np.percentile(spectral_index_global, 1), np.percentile(spectral_index_global, 99))
plt.ylim(np.percentile(correction_factor_global, 1), np.percentile(correction_factor_global, 99))
plt.ylabel("Correction factor")
plt.xlabel(r"Fitted spectral index $\alpha$")
plt.title("Correction factor as function of fitted spectral index\nall catalogs")
plt.show()

print("------------------------------------------------")
print(f"Spectral index: {round(px,3)}, correction factor: {round(py,3)}")

#### position dependant correction factor
# plt.scatter(ras[o], decs[o], c=correction_factor_global[o], norm='log')
# plt.colorbar()
# plt.ylabel("DEC (deg)")
# plt.xlabel("RA (deg)")
# plt.show()

#### variables
ras, decs = [], []              # positional coordinates
correction_factor_global = []   # ratio between read-out catalog0 flux and computed
spectral_index_global = []      # per-source two-point spectral index
fitted_flux = []                # catalog0 flux based on two-point spectral index extrapolation
signal_to_noise = []            # signal-to-noise (flux_jy / e_flux_jy)
catalog_weight_factor = []      # weighting factor based on systematic catalog uncertainty; to be deprecated
max_separation = []             # maximum per-source separation between all three matched catalog positions
point_probability = []          # probability of points matching

###################################################
#### catalog four-way combination auto-looper ####
###################################################
# for combination in get_combinations(catalogs, size=4, required_index=6, skip_index=2):
#     local_cats = [catalogs[i] for i in combination]

#     output = compute_flux_correction_factor(local_cats, default_config, debug=debug)

#     if output is not None:
#         spx, snr, cor, flux, catw, max_sep, p_weight, ra, dec = output

#         ras += [ra]; decs += [dec]
#         correction_factor_global += [cor]
#         spectral_index_global += [spx]
#         fitted_flux += [flux]
#         signal_to_noise += [snr]
#         catalog_weight_factor += [catw]
#         max_separation += [max_sep]
#         point_probability += [p_weight]