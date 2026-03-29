from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from functions import match_catalogs_2D, compute_fluxcal_statistics, get_spectral_index, calculate_contour_statistics, get_combinations, get_pos_err_deg, weighted_bin_stats, weighted_bin_stats_2d, sources_in_fits
from astropy.table import Table
from scipy.stats import chi2
from itertools import combinations
from time import perf_counter

from catalog_manager import catalog, config, catalog_set

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
def compute_flux_correction_factor(cats, config, debug=False, internal_output=False):
    indices, quality = match_catalogs_2D(cats, thres_arc=config.thres_arc, return_quality=True, nsigma=config.nsigma, thres_arc_override=config.thres_arc_override, crowd_radius_arc=config.crowd_radius_arc)

    # if there are too few sources, return None
    if len(indices[0]) <= config.minimum_points:
        if internal_output: print(f"Error: no source-matches found between [{', '.join(f'{cat.name}' for cat in cats)}]")
        return None

    # figure out for which catalog the correction factor should be calculated
    # if cat is given, match that, otherwise assume the first index
    other_index = [i for i in range(len(cats))]
    if config.anchor_catalog is None:
        anchor_index = 0
    else:
        try:
            anchor_index = cats.index(config.anchor_catalog)
        except:
            print("Anchor catalog missing from inputs")
            return None
    other_index.remove(anchor_index)

    # if crowding parameter is available, store max neighbour (withing crowding_radius) count per source
    n_crowd = np.zeros(len(indices[0]), dtype=int)
    if quality['n_crowd']:
        for cat_idx, match_idx in enumerate(indices):
            nc = quality['n_crowd'].get(cat_idx)
            if nc is not None and len(nc):
                n_crowd = np.maximum(n_crowd, nc[match_idx])
    
    # re-check after crowding
    if len(indices[0]) <= config.minimum_points:
        if internal_output: print(f"Error: no source-matches found between [{', '.join(f'{cat.name}' for cat in cats)}] after crowding check")
        return None

    # create subsets of all catalogs, such that we can ignore (i0,i1,...) afterwards
    for index, cat in enumerate(cats): cats[index] = cat.create_subset(indices[index])

    # all pairwise separations, then take per-source maximum
    coords = [SkyCoord(ra=cat.ra * u.deg, dec=cat.dec * u.deg) for cat in cats]
    sig    = [np.rad2deg(cat.err_rad) * 3600 for cat in cats]   # arcsec per-source

    pair_seps = {}
    p_weight  = np.ones(len(cats[0].ra))
    for a, b in combinations(range(len(cats)), 2):
        sep = coords[a].separation(coords[b]).arcsec
        pair_seps[(a, b)] = sep
        p_weight *= 1.0 - chi2.cdf((sep / np.hypot(sig[a], sig[b])) ** 2, df=2)

    max_sep = np.maximum.reduce(list(pair_seps.values()))

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
        plt.xlabel(f"{cats[anchor_index].name} linear flux (Jy)")
        plt.ylabel(f"{cats[anchor_index].name} fitted flux (Jy)")
        plt.title(f"{cats[anchor_index].name} "+r"flux, $\alpha$=-0.7 vs fitted")
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
        
    if internal_output: print(f"Completed set [{', '.join(f'{cat.name:9}' for cat in cats)}]", f"Matches: {len(indices[0])}")    
    
    return (spectral_indices, snr, correction_factor, extrapolated_flux_fit, catalog_weight_factor, max_sep, p_weight, n_crowd, ra, dec)

"""Calculate weighted correction factor based on per-point spectral indices, signal-to-noise, and correction factor"""
def calculate_correction_factor_weight(spx, snr, catw, max_sep, p_match, n_crowd, config):
    # downweight sources with spectral indices far away from -0.7
    spectral_difference_factor = np.exp(-config.spectral_damping_factor * (spx - config.spectral_index_theory)**2)
    
    # discard low snr sources
    signal_to_noise_factor = snr.copy()
    signal_to_noise_factor[signal_to_noise_factor < config.snr_lower_limit] = 0
    
    # weighting based on separation between points
    separation_weight = np.exp(-(max_sep / config.thres_arc) ** 2)
    
    return spectral_difference_factor * signal_to_noise_factor * catw * separation_weight * p_match

"""Extrapolate flux based on two frequencies, one flux, and a spectral index"""
def get_flux_from_index(spectral_index, reference_flux, current_frequency, reference_frequency):
    return reference_flux * (current_frequency / reference_frequency) ** spectral_index

start = perf_counter()

#### all available catalogs
all_catalogs = catalog_set([
    catalog("/catalogs/racs/racs_gal_clean.fits",             887.5e6,    "racs_gal"),  # the galactic portion of the racs survey
    catalog("/catalogs/racs/racs_full_clean.fits",            887.5e6,    "racs"),      # the rest of the racs survey
    catalog("/catalogs/meerkat/meerkat_clean.fits",           1359.7e6,   "meerkat"),
    catalog("/catalogs/vlssr/vlssr_clean.fits",               73.8e6,     "vlssr"),
    catalog("/catalogs/tgss/tgss_clean.fits",                 150e6,      "tgss"),
    catalog("/catalogs/gleam_300/gleam_300_clean.fits",       300e6,      "gleam_300"),
    catalog("/catalogs/gleam_x_gp/gleam_x_gp_clean.fits",     200e6,      "gleam_xgp"),
    catalog("/catalogs/nvss/nvss_clean.fits",                 1400e6,     "nvss"),
    catalog("/catalogs/wenss/wenss_clean.fits",               325e6,      "wenss"),
    catalog("/catalogs/lofar/lofar_sources_pipeline.fits",    144.6e6,    "lofar"),     # LOFAR P282+00
    catalog("/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",  144.6e6,    "lofar_dr3"),
    catalog("/catalogs/other/cygnus_clean.fits",              336e6,      "cygnus"),    # vla cygnus region
    ])

racs_gal, racs, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar, lofar_dr3, cygnus = all_catalogs.catalogs

#### available configurations
lofar_dr3_config = config(spectral_damping_factor = 5,
                          snr_lower_limit = 7,
                          nsigma = 2,
                          minimum_points = 3,
                          crowd_radius_arc = 10,
                          minimum_frequency_spacing = None,
                          catalogs = [racs, racs_gal, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar_dr3],
                          reference_file = None,
                          anchor_catalog = lofar_dr3,
                          )

default_config = config(spectral_damping_factor = 5,
                        snr_lower_limit = 7,
                        nsigma = 3,
                        minimum_points = 3,
                        crowd_radius_arc = None,
                        minimum_frequency_spacing = None,
                        catalogs = [racs_gal, meerkat, tgss, gleam_300, gleam_xgp, lofar],
                        reference_file = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0],
                        anchor_catalog = lofar,
                        )

cygnus_config = config(spectral_damping_factor = 5,
                       snr_lower_limit = 7,
                       nsigma = 3,
                       minimum_points = 3,
                       crowd_radius_arc = None,
                       minimum_frequency_spacing = None,
                       catalogs = [racs, vlssr, tgss, gleam_300, nvss, wenss, lofar_dr3, cygnus],
                       reference_file = np.sort(glob.glob(os.getcwd()+"/data/other/*.fits"))[0],
                       anchor_catalog = cygnus,
                       )

small_config = config(spectral_damping_factor = 5,
                       snr_lower_limit = 7,
                       nsigma = 3,
                       minimum_points = 3,
                       crowd_radius_arc = None,
                       minimum_frequency_spacing = None,
                       catalogs = [lofar_dr3, nvss, racs],
                       reference_file = None,
                       anchor_catalog = racs,
                       )

#### Parameters
debug = False
#config = lofar_dr3_config
#config = default_config
config = cygnus_config
#config = small_config

config.setup()

print(f"Setup done at: {perf_counter() - start} s")

if debug:    
    # cutdown catalog plot
    for cat in config.catalogs:
        plt.hist(np.log10(cat.flux), alpha=0.6, bins=25, label=cat.name)
    plt.xlabel("log10(flux/Jy)")
    plt.ylabel("count")
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    # catalog as function of position
    for cat in config.catalogs:
        if len(cat.ra) > 0: plt.scatter(cat.ra, cat.dec, s=1, label=cat.name)
    plt.gca().set_box_aspect(1)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend(loc='lower left')
    plt.show()

#### variables
ras, decs = [], []              # positional coordinates
correction_factor_global = []   # ratio between read-out catalog0 flux and computed
spectral_index_global = []      # per-source two-point spectral index
fitted_flux = []                # catalog0 flux based on two-point spectral index extrapolation
signal_to_noise = []            # signal-to-noise (flux_jy / e_flux_jy)
catalog_weight_factor = []      # weighting factor based on systematic catalog uncertainty; to be deprecated
max_separation = []             # maximum per-source separation between all three matched catalog positions
point_probability = []          # probability of points matching
crowding_parameter = []         # maximum number of neighbours per source within crowd_radius_arc

###################################################
#### catalog three-way combination auto-looper ####
###################################################
all_combinations = get_combinations(config.catalogs, size=3, required_index=config.anchor_catalog_index)
output_width = len(str(len(all_combinations)))
for i, combination in enumerate(all_combinations):
    local_cats = [config.catalogs[j] for j in combination]
    output = compute_flux_correction_factor(local_cats, config, debug=debug)
    
    if output is not None:
        spx, snr, cor, flux, catw, max_sep, p_weight, n_crowd, ra, dec = output
        print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", f"Matches: {len(spx)}")
        
        ras += [ra]; decs += [dec]
        correction_factor_global += [cor]
        spectral_index_global += [spx]
        fitted_flux += [flux]
        signal_to_noise += [snr]
        catalog_weight_factor += [catw]
        max_separation += [max_sep]
        point_probability += [p_weight]
        crowding_parameter += [n_crowd]
    else:
        print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", "Matches: None")

print(f"Calculations done at: {perf_counter() - start} s")

###########################################################
#### plotting correction factor for each catalog combo ####
###########################################################
# for spx, snr, cor, catw, max_sep in zip(spectral_index_global, signal_to_noise, correction_factor_global, catalog_weight_factor, max_separation):
#     total_weighting_factor = calculate_weighted_correction_factor(spx, snr, catw, max_sep, config)
#     Xi, Yi, Zi, px, py = calculate_contour_statistics(spx, cor, total_weighting_factor, logy=True)

#     o = np.argsort(total_weighting_factor)

#     fig, ax = plt.subplots()
#     sc = ax.scatter(spx[o], cor[o], c=total_weighting_factor[o])
#     ax.contour(Xi, Yi, Zi, levels=6, colors='red', alpha=0.7, linewidths=0.8)
#     plt.colorbar(sc, label="Combined weighting factor")
#     plt.yscale('log')

#     plt.axhline(py, ls="--", color="gray")
#     plt.axvline(px, ls="--", color="gray")

#     plt.xlim(np.percentile(spx, 1), np.percentile(spx, 99))
#     plt.ylim(np.percentile(cor, 1), np.percentile(cor, 99))
#     plt.ylabel("Correction factor")
#     plt.xlabel(r"Fitted spectral index $\alpha$")
#     plt.title("Correction factor as function of fitted spectral index")
#     plt.show()

#     print(f"Spectral index: {round(px,3)}, correction factor: {round(py,3)}")

ras = np.concatenate(ras)
decs = np.concatenate(decs)
correction_factor_global = np.concatenate(correction_factor_global)
spectral_index_global = np.concatenate(spectral_index_global)
fitted_flux = np.concatenate(fitted_flux)
signal_to_noise = np.concatenate(signal_to_noise)
catalog_weight_factor = np.concatenate(catalog_weight_factor)
max_separation = np.concatenate(max_separation)
crowding_parameter = np.concatenate(crowding_parameter)
point_probability = np.concatenate(point_probability)

total_weighting_factor = calculate_correction_factor_weight(spectral_index_global,
                                                            signal_to_noise,
                                                            catalog_weight_factor,
                                                            max_separation,
                                                            point_probability,
                                                            crowding_parameter,
                                                            config)


############################################################################
#### plotting correction factor based on all previous catalog matchings ####
############################################################################
Xi, Yi, Zi, px, py = calculate_contour_statistics(spectral_index_global, correction_factor_global, total_weighting_factor, logy=True, n=1000)

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
print(f"Spectral index: {round(px,3)}, correction factor: {round(py,3)}, total matches: {len(correction_factor_global)}")


##########################
#### inspection plots ####
##########################

#### position dependant correction factor
#o = np.argsort(correction_factor_global)
f = (correction_factor_global[o] > 1e-2) & (correction_factor_global[o] < 1e2)
plt.scatter(ras[o][f], decs[o][f], c=correction_factor_global[o][f], s=2, norm='log')
plt.colorbar(label='Correction factor')
plt.ylabel("DEC (deg)")
plt.xlabel("RA (deg)")
plt.show()

#### correction factor as function of total weighting factor
plt.scatter(total_weighting_factor, correction_factor_global, s=1.5, alpha=0.2)
plt.yscale('log')
plt.xscale('log')
plt.axhline(1, ls='--', color='black', alpha=0.5, label='1')
plt.axhline(py, ls='--', color='tomato', label='Fit')
plt.ylabel("Correction factor")
plt.xlabel("Total weighting factor")
plt.legend()
plt.show()

#### correction factor as function of ra and dec separately
mask = (correction_factor_global < 10) & (correction_factor_global > 0.1)

cmean = np.average(correction_factor_global[mask], weights=total_weighting_factor[mask])
cstd = np.std(correction_factor_global[mask])
cmin, cmax = max(0.1, cmean - 3 * cstd), cmean + 3 * cstd
mask &= (correction_factor_global > cmin) & (correction_factor_global < cmax) & (total_weighting_factor > 0)

dec_c, dec_mn, dec_std = weighted_bin_stats(decs[mask], correction_factor_global[mask], total_weighting_factor[mask], n_bins=50)
ra_c,  ra_mn,  ra_std  = weighted_bin_stats(ras[mask],  correction_factor_global[mask], total_weighting_factor[mask], n_bins=50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(dec_c, dec_mn, color='steelblue', lw=2, label='Weighted mean')
ax1.fill_between(dec_c, dec_mn - dec_std, dec_mn + dec_std, alpha=0.25, color='steelblue', label='±1σ (weighted)')
ax1.set_xlabel('Dec (deg)')
ax1.set_ylabel('Correction factor')
ax1.legend()
ax2.plot(ra_c, ra_mn, color='tomato', lw=2, label='Weighted mean')
ax2.fill_between(ra_c, ra_mn - ra_std, ra_mn + ra_std, alpha=0.25, color='tomato', label='±1σ (weighted)')
ax2.set_xlabel('RA (deg)')
ax2.legend()
fig.suptitle('Weighted correction factor')
plt.tight_layout()
plt.show()

#### correction factor as function of [ra, dec] in 2D
ra_c2, dec_c2, wmean_2d, wstd_2d = weighted_bin_stats_2d(
    ras[mask], decs[mask],
    correction_factor_global[mask],
    total_weighting_factor[mask],
    n_bins=40, m_bins=40
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

im1 = ax1.pcolormesh(ra_c2, dec_c2, wmean_2d.T, cmap='RdYlGn_r', shading='auto')
fig.colorbar(im1, ax=ax1, label='Correction factor')
ax1.set_xlabel('RA (deg)')
ax1.set_ylabel('Dec (deg)')
ax1.set_title('Weighted mean')

im2 = ax2.pcolormesh(ra_c2, dec_c2, wstd_2d.T, cmap='plasma', shading='auto')
fig.colorbar(im2, ax=ax2, label='Std')
ax2.set_xlabel('RA (deg)')
ax2.set_ylabel('Dec (deg)')
ax2.set_title('Weighted ±1σ')

fig.suptitle('Correction factor map')
plt.tight_layout()
plt.show()

print(f"Done at: {perf_counter() - start} s")

#### plot point densities per catalog
# for cat in config.catalogs:
#     plt.hist2d(-cat.ra, cat.dec, bins=(400, 100))
#     plt.title(cat.name)
#     plt.show()

#### variables
# ras, decs = [], []              # positional coordinates
# correction_factor_global = []   # ratio between read-out catalog0 flux and computed
# spectral_index_global = []      # per-source two-point spectral index
# fitted_flux = []                # catalog0 flux based on two-point spectral index extrapolation
# signal_to_noise = []            # signal-to-noise (flux_jy / e_flux_jy)
# catalog_weight_factor = []      # weighting factor based on systematic catalog uncertainty; to be deprecated
# max_separation = []             # maximum per-source separation between all three matched catalog positions
# point_probability = []          # probability of points matching

###################################################
#### catalog four-way combination auto-looper ####
###################################################
# for combination in get_combinations(catalogs, size=4, required_index=6, skip_index=2):
#     local_cats = [catalogs[i] for i in combination]

#     output = compute_flux_correction_factor(local_cats, config, debug=debug)

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