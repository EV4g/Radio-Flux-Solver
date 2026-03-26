from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import copy
from functions import match_catalogs_2D, compute_fluxcal_statistics, get_spectral_index, calculate_contour_statistics, get_triplet_combinations, get_pos_err
from astropy.table import Table
from scipy.stats import chi2

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
def compute_flux_correction_factor(cats, config, debug=False):
    (i0, i1, i2), quality = match_catalogs_2D(cats, thres_arc=config.thres_arc, return_quality=True, nsigma=config.nsigma)

    # if there are no sources, return
    if len(i0) <= 1:
        if debug: print(f"Error: no source-matches found between {cats[0].name}, {cats[1].name}, {cats[2].name}")
        return None

    # if crowding parameter is available, remove overcrowded points; else skip
    if quality['n_crowd']:
        crowd_ok = np.ones(len(i0), dtype=bool)
        for cat_idx, match_idx in zip([0, 1, 2], [i0, i1, i2]):
            nc = quality['n_crowd'].get(cat_idx)
            if nc is not None and len(nc): crowd_ok &= (nc[match_idx] == 0)
        i0, i1, i2 = i0[crowd_ok], i1[crowd_ok], i2[crowd_ok]

    # create subsets of all catalogs, such that we can ignore (i0,i1,i2) afterwards
    for index, cat in enumerate(cats): cats[index] = cat.create_subset([i0, i1, i2][index])

    # per source maximum separation
    coords = [SkyCoord(ra=cat.ra * u.deg, dec=cat.dec * u.deg) for cat in cats]
    sep_01  = coords[0].separation(coords[1]).arcsec
    sep_02  = coords[0].separation(coords[2]).arcsec
    sep_12  = coords[1].separation(coords[2]).arcsec
    max_sep = np.maximum(sep_01, np.maximum(sep_02, sep_12))

    # point probability weighting
    sig = [get_pos_err(cat)*3600 for cat in cats]   # arcsec per-source
    p01 = 1.0 - chi2.cdf((sep_01 / np.hypot(sig[0], sig[1])) ** 2, df=2)
    p02 = 1.0 - chi2.cdf((sep_02 / np.hypot(sig[0], sig[2])) ** 2, df=2)
    p12 = 1.0 - chi2.cdf((sep_12 / np.hypot(sig[1], sig[2])) ** 2, df=2)
    p_weight = p01 * p02 * p12

    # flux and flux error
    uncorrected_flux = cats[0].flux
    uncorrected_flux_error = cats[0].e_flux

    # calculate two-point spectral index
    spectral_indices = get_spectral_index(cats[1].flux, cats[2].flux, cats[1].freq, cats[2].freq)

    # fit is based on measuring between two points, linear is average between assuming theoretical extragalactic value for both
    extrapolated_flux_fit = get_flux_from_index(spectral_indices, cats[1].flux, cats[0].freq, cats[1].freq)
    extrapolated_flux_linear = 0.5 * (get_flux_from_index(config.spectral_index_theory, cats[1].flux, cats[0].freq, cats[1].freq)
                                    + get_flux_from_index(config.spectral_index_theory, cats[2].flux, cats[0].freq, cats[2].freq))

    if debug:
        for flux in zip(cats[0].flux, cats[1].flux, cats[2].flux):
            plt.plot(np.array([cats[0].freq, cats[1].freq, cats[2].freq])*1e-6, flux)

        plt.yscale('log')
        plt.ylabel(r"log$_{10}$(Jy)")
        plt.xlabel("Frequency (MHz)")
        plt.show()

    correction_factor = extrapolated_flux_fit / uncorrected_flux
    snr = uncorrected_flux / uncorrected_flux_error

    catalog_weight_factor = np.ones_like(snr)
    for cat in cats:
        if cat.name.lower() == "tgss": catalog_weight_factor *= 0.5
        if cat.name.lower() == "gleam_300": catalog_weight_factor *= 0.7
        if cat.name.lower() == "gleam_xgp": catalog_weight_factor *= 0.8

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

    print(f"Completed set [{cats[0].name:9}, {cats[1].name:9}, {cats[2].name:9}]", round(catalog_weight_factor[0],2) if debug else "")

    return (spectral_indices, snr, correction_factor, extrapolated_flux_fit, catalog_weight_factor, max_sep, p_weight, cats[0].ra, cats[0].dec)

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
default_config = config(thres_arc=2, spectral_damping_factor=5, snr_lower_limit=7)
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
        plt.scatter(cat.ra, cat.dec, s=1, label=cat.name)
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
for combination in get_triplet_combinations(catalogs, required_index=6, skip_index=2):
    local_cats  = [catalogs[combination[0]], catalogs[combination[1]], catalogs[combination[2]]]

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

#### Quad matchings below

#################################
# lofar + tgss + racs + meerkat #
#################################
# i1, i2, i3, i4 =  match_catalogs_2D((lofar, tgss, racs, meerkat))
# freqs = np.array([lofar_freq, tgss_freq, racs_freq, meerkat_freq]) * 1e-6
# spectral_indices = []             # fitted spectral index based on racs, meerkat
# extrapolated_flux_linear = []     # extrapolated flux when assuming a simple -0.7
# extrapolated_flux_fit = []        # extrapolated flux when fitting the fluxes from racs and meerkat
# lofar_uncorrected_flux = []       # current lofar flux, no corrections
# lofar_uncorrected_flux_error = [] # current lofar flux error, no corrections

# for i, (lofar_i, tgss_i, racs_i, meerkat_i) in enumerate(zip(i1, i2, i3, i4)):
#     lofar_F, tgss_F, racs_F, meerkat_F = lofar['flux_jy'][lofar_i], tgss['flux_jy'][tgss_i], racs['flux_jy'][racs_i], meerkat['flux_jy'][meerkat_i]
#     #spectral_index = get_spectral_index(tgss_F, racs_F, freqs[1], freqs[2])
#     #spectral_indices.append(spectral_index)

#     lofar_uncorrected_flux.append(lofar_F)
#     lofar_uncorrected_flux_error.append(lofar['e_flux_jy'][lofar_i])

#     # lofar flux when using racs flux and extrapolating back to lofar freq
#     #extrapolated_flux_fit.append(get_flux_from_index(spectral_index, tgss_F, freqs[0], freqs[1]))
#     #extrapolated_flux_linear.append(0.5 * (get_flux_from_index(-0.7, tgss_F, freqs[0], freqs[1]) + get_flux_from_index(-0.7, racs_F, freqs[0], freqs[2])))


#     plt.plot(freqs, [lofar_F, tgss_F, racs_F, meerkat_F])

# plt.yscale('log')
# plt.ylabel(r"log$_{10}$(Jy)")
# plt.xlabel("Frequency (MHz)")
# plt.show()

# extrapolated_flux_fit = np.array(extrapolated_flux_fit)
# lofar_uncorrected_flux = np.array(lofar_uncorrected_flux)
# lofar_uncorrected_flux_error = np.array(lofar_uncorrected_flux_error)
# spectral_indices = np.array(spectral_indices)

# valid_factor = (spectral_indices > -1) & (spectral_indices < 0)
# correction_factor = extrapolated_flux_fit / lofar_uncorrected_flux


# # compare -0.7 assumption versus fitted spectral indices
# mn, mx = min(np.min(extrapolated_flux_fit), np.min(extrapolated_flux_linear)), max(np.max(extrapolated_flux_fit), np.max(extrapolated_flux_linear))
# plt.scatter(extrapolated_flux_linear, extrapolated_flux_fit, c=spectral_indices)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(mn, mx)
# plt.ylim(mn, mx)
# plt.gca().set_box_aspect(1)
# plt.plot((mn, mx), (mn, mx), c='k', ls='--')
# plt.colorbar(label = r"Spectral index $\alpha$")
# plt.xlabel("Lofar linear flux (Jy)")
# plt.ylabel("Lofar fitted flux (Jy)")
# plt.title(r"Lofar flux, $\alpha$=-0.7 vs fitted")
# plt.show()

# # compare fitted spectral index with correction factor
# plt.scatter(spectral_indices, correction_factor, c=lofar_uncorrected_flux, norm='log')
# plt.yscale('log')
# plt.axvline(-0.7, ls='--', c='k')
# plt.axhline(1, ls='--', c='k')
# plt.colorbar(label='Flux (Jy)')
# plt.ylabel("Flux correction factor")
# plt.xlabel(r"Spectral index $\alpha$")
# plt.title("Flux correction as function of spectral index")
# plt.show()

# # signal to noise
# snr = lofar_uncorrected_flux / lofar_uncorrected_flux_error

# # distance of spectral index to alpha = -0.7
# spectral_difference = np.abs(spectral_indices + 0.7)
# spectral_difference_factor = (1 - (spectral_difference / np.max(spectral_difference)))**2

# # logarithmic weighted mean of the flux correction factor
# weighted_mean_correction = 10**(np.mean(snr * spectral_difference_factor * np.log10(correction_factor)) / np.mean(snr * spectral_difference_factor))
