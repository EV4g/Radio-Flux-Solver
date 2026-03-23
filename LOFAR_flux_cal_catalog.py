from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import multiprocessing; multiprocessing.set_start_method('fork') #for windows/mac
from functions import match_catalogs_2D, compute_fluxcal_statistics, get_spectral_index, calculate_contour_statistics
from astropy.table import Table
from itertools import combinations

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
    idx_cat1, idx_cat2 = match_catalogs_2D([
        (cat1["ra"], cat1["dec"]),
        (cat2["ra"], cat2["dec"])], thres_arc=thres_arc)

    flux1 = cat1['flux_jy'][idx_cat1]
    snr1 = flux1 / cat1['e_flux_jy'][idx_cat1]

    flux2 = cat2['flux_jy'][idx_cat2]
    snr2 = flux2 / cat2['e_flux_jy'][idx_cat2]
    
    return flux1, flux2, snr1, snr2

"""Load two catalogs and plot their relative fluxes, according to a powerlaw"""
def quick_compare_catalog(cat1, cat2, freq1, freq2, name1, name2, thres_arc=2, spectral_index_theory=-0.7):
    flux1, flux2, _, _ = get_catalog_matched_flux(cat1, cat2, thres_arc=thres_arc)
    spectral_flux_ratio, spectral_index_actual, x, log_ratio, scale_factor = compute_fluxcal_statistics(freq1, freq2, flux1, flux2, spectral_index_theory)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(flux2, flux1, s=10, alpha=0.7)
    ax[0].plot(x, x * spectral_index_actual, color='purple', ls='--', label="Data fit")
    ax[0].plot(x, x, color='black', ls='--', label="x = y")
    ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={spectral_index_theory})')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(f"{name1} flux [Jy]"); ax[0].set_ylabel(f"{name2} flux [Jy]")
    ax[0].legend()

    # Log-ratio histogram
    ax[1].hist(log_ratio, bins=30, edgecolor='k')
    ax[1].axvline(scale_factor, color='red', ls='--', label=f'Median = {scale_factor:.3f}')
    ax[1].set_xlabel(f"log10(S_{name1} / S_{name2})")
    ax[1].set_ylabel("N")
    ax[1].legend()
    plt.tight_layout()
    #plt.savefig("flux_scale_comparison.png", dpi=150)
    plt.show()
    
    print(f"Compared {name1} to {name2} \n")

"""compute the flux correction factor based on three given catalogs. Catalogs are matches, and the last two are used to calculate the spectral index
which is used to extrapolate what the first cat -should- be. The different between -should- and -is-, is the correction factor."""
def compute_flux_correction_factor(cats, freqs, names, debug=False, thres_arc=2, return_coord=False):
    i1, i2, i3 =  match_catalogs_2D(radec_list(cats), thres_arc=thres_arc)
    spectral_indices = []             # fitted spectral index based on racs, meerkat
    extrapolated_flux_linear = []     # extrapolated flux when assuming a simple -0.7
    extrapolated_flux_fit = []        # extrapolated flux when fitting the fluxes from racs and meerkat
    lofar_uncorrected_flux = []       # current lofar flux, no corrections
    lofar_uncorrected_flux_error = [] # current lofar flux error, no corrections
    
    # if there are no sources, return
    if len(i1) <= 1: 
        print("Error: no source-matches found between selected catalogs")
        return
    
    for i, (ai, bi, ci) in enumerate(zip(i1, i2, i3)):
        fluxes = [cats[0]['flux_jy'][ai], cats[1]['flux_jy'][bi], cats[2]['flux_jy'][ci]]
        e_fluxes = [cats[0]['e_flux_jy'][ai], cats[1]['e_flux_jy'][bi], cats[2]['e_flux_jy'][ci]]
        spectral_index = get_spectral_index(fluxes[1], fluxes[2], freqs[1], freqs[2])
        spectral_indices.append(spectral_index)
        
        lofar_uncorrected_flux.append(fluxes[0])
        lofar_uncorrected_flux_error.append(e_fluxes[0])
        
        # lofar flux when using racs flux and extrapolating back to lofar freq
        extrapolated_flux_fit.append(get_flux_from_index(spectral_index, fluxes[1], freqs[0], freqs[1]))
        extrapolated_flux_linear.append(0.5 * (get_flux_from_index(-0.7, fluxes[1], freqs[0], freqs[1]) + get_flux_from_index(-0.7, fluxes[2], freqs[0], freqs[2])))
        
        
        if debug: plt.plot(np.array(freqs)*1e-6, fluxes)
    
    if debug:
        plt.yscale('log')
        plt.ylabel(r"log$_{10}$(Jy)")
        plt.xlabel("Frequency (MHz)")
        plt.show()

    extrapolated_flux_fit = np.array(extrapolated_flux_fit)
    extrapolated_flux_linear = np.array(extrapolated_flux_linear)
    extrapolated_flux_fit_to_linear_ratio = extrapolated_flux_fit / extrapolated_flux_linear
    
    
    lofar_uncorrected_flux = np.array(lofar_uncorrected_flux)
    lofar_uncorrected_flux_error = np.array(lofar_uncorrected_flux_error)
    spectral_indices = np.array(spectral_indices)

    correction_factor = extrapolated_flux_fit / lofar_uncorrected_flux
    snr = lofar_uncorrected_flux / lofar_uncorrected_flux_error
    
    catalog_weight_factor = np.ones_like(snr)
    for name in names:
        if name.lower() == "tgss": catalog_weight_factor *= 0.5
        if name.lower() == "gleam_300": catalog_weight_factor *= 0.7
        if name.lower() == "gleam_xgp": catalog_weight_factor *= 0.8
    
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
        plt.xlabel(f"{names[0]} linear flux (Jy)")
        plt.ylabel(f"{names[0]} fitted flux (Jy)")
        plt.title(f"{names[0]} "+r"flux, $\alpha$=-0.7 vs fitted")
        plt.show()
    
        # compare fitted spectral index with correction factor
        plt.scatter(spectral_indices, correction_factor, c=lofar_uncorrected_flux, norm='log')
        plt.yscale('log')
        plt.axvline(-0.7, ls='--', c='k')
        plt.axhline(1, ls='--', c='k')
        plt.colorbar(label='Flux (Jy)')
        plt.ylabel("Flux correction factor")
        plt.xlabel(r"Spectral index $\alpha$")
        plt.title("Flux correction as function of spectral index")
        plt.show()
        
    print(f"Completed set [{names[0]:9}, {names[1]:9}, {names[2]:9}]", round(catalog_weight_factor[0],2) if debug else "")
        
    if return_coord:
        return spectral_indices, snr, correction_factor, extrapolated_flux_fit, extrapolated_flux_fit_to_linear_ratio, catalog_weight_factor, cats[0]['ra'][i1], cats[0]['dec'][i1]
    else:
        return spectral_indices, snr, correction_factor, extrapolated_flux_fit, extrapolated_flux_fit_to_linear_ratio, catalog_weight_factor

"""Calculate weighted correction factor based on per-point spectral indices, signal-to-noise, and correction factor"""
def calculate_weighted_correction_factor(spx, snr, cor, catw, spx_ref=-0.7, spectral_damping_factor=10):
    spectral_difference_factor = np.exp(-spectral_damping_factor * (spx - spx_ref)**2)
    signal_to_noise_factor = snr
    
    return spectral_difference_factor * signal_to_noise_factor * catw

"""(cat1, cat2) --> [(ra1, dec1), (ra2, dec2)]"""
def radec_list(cats):
    radec_list = []
    for cat in cats:
        radec_list.append((cat['ra'], cat['dec']))
    return radec_list        

"""Return indices of catalog (str) of all unique, non-double, threeway combinations with the condition f1 < f2 < f3"""
def get_triplet_combinations(frequencies, catalogs, required_index=None, skip_index=None):
    indexed = sorted(enumerate(zip(frequencies, catalogs)), key=lambda x: x[1][0])
    return [(i1, i2, i3) for (i1, (f1, _)), (i2, (f2, _)), (i3, (f3, _)) in combinations(indexed, 3) if f1 < f2 < f3 
        and (required_index is None or required_index in (i1, i2, i3)) and (skip_index is None or skip_index not in (i1, i2, i3))]

"""Extrapolate flux based on two frequencies, one flux, and a spectral index"""
def get_flux_from_index(spectral_index, reference_flux, current_frequency, reference_frequency):
    return reference_flux * (current_frequency / reference_frequency) ** spectral_index

# lofar fits file
lofar_files = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0]

# get calagogs, names in order of acquisition, lofar last
survey_names = np.array(["racs", "meerkat", "vlssr", "tgss", "gleam_300", "gleam_xgp", "lofar"])

racs_full      = Table.read(os.getcwd()+"/catalogs/racs/racs_clean.csv")
meerkat_full   = Table.read(os.getcwd()+"/catalogs/meerkat/meerkat_clean.csv")
vlssr_full     = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_clean.csv")
tgss_full      = Table.read(os.getcwd()+"/catalogs/tgss/tgss_clean.fits")
gleam_300_full = Table.read(os.getcwd()+"/catalogs/gleam_300/gleam_300_clean.fits")
gleam_xgp_full = Table.read(os.getcwd()+"/catalogs/gleam_x_gp/gleam_x_gp_clean.fits")
lofar_full     = Table.read(os.getcwd()+"/catalogs/lofar/lofar_sources_pipeline.fits")
#lofar_full    = Table.read(os.getcwd()+"/catalogs/lofar/LoTSS_DR3_v1.0.srl.fits")

spectral_index_theory = -0.7
lofar_freq     = 144.6e6  #Hz
racs_freq      = 887.5e6  #Hz
meerkat_freq   = 1359.7e6 #Hz
vlssr_freq     = 73.8e6   #Hz
tgss_freq      = 150e6    #Hz
gleam_300_freq = 300e6    #Hz
gleam_xgp_freq = 200e6    #Hz

survey_frequencies = np.array([racs_freq, meerkat_freq, vlssr_freq, tgss_freq, gleam_300_freq, gleam_xgp_freq, lofar_freq])

# fix lofar
lofar_full.rename_column("RA", "ra")
lofar_full.rename_column("DEC", "dec")
lofar_full.rename_column('Total_flux', 'flux_jy')
lofar_full.rename_column('E_Total_flux', 'e_flux_jy')

# LOFAR-DR3 uses mJy, pybdsf uses Jy, force everything to Jy
if str(lofar_full['flux_jy'].unit) == 'mJy':
    lofar_full['flux_jy'] *= 1e-3
    lofar_full['e_flux_jy'] *= 1e-3

# check for sources in current file
racs_valid      = sources_in_fits(racs_full['ra'],      racs_full['dec'],       lofar_files)
meerkat_valid   = sources_in_fits(meerkat_full['ra'],   meerkat_full['dec'],    lofar_files)
vlssr_valid     = sources_in_fits(vlssr_full['ra'],     vlssr_full['dec'],      lofar_files)
tgss_valid      = sources_in_fits(tgss_full['ra'],      tgss_full['dec'],       lofar_files)
gleam_300_valid = sources_in_fits(gleam_300_full['ra'], gleam_300_full['dec'],  lofar_files)
gleam_xgp_valid = sources_in_fits(gleam_xgp_full['ra'], gleam_xgp_full['dec'],  lofar_files)
lofar_valid     = sources_in_fits(lofar_full['ra'],     lofar_full['dec'],      lofar_files)

# remove all non-valid points to reduce syntax clutter
racs      = racs_full[racs_valid]
meerkat   = meerkat_full[meerkat_valid]
vlssr     = vlssr_full[vlssr_valid]
tgss      = tgss_full[tgss_valid]
gleam_300 = gleam_300_full[gleam_300_valid]
gleam_xgp = gleam_xgp_full[gleam_xgp_valid]
lofar     = lofar_full[lofar_valid]

catalogs      = [racs, meerkat, vlssr, tgss, gleam_300, gleam_xgp, lofar]
catalogs_full = [racs_full, meerkat_full, vlssr_full, tgss_full, gleam_300_full, gleam_xgp_full, lofar_full]

# full catalog plot
for catalog, name in zip(catalogs_full, survey_names):
    plt.hist(np.log10(catalog['flux_jy']), alpha=0.6, bins=75, label=name)
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.legend()
plt.show()

# cutdown catalog plot
for catalog, name in zip(catalogs, survey_names):
    plt.hist(np.log10(catalog['flux_jy']), alpha=0.6, bins=25, label=name)
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.legend()
plt.show()

# catalog as function of position
for catalog, name in zip(catalogs, survey_names):
    plt.scatter(catalog['ra'], catalog['dec'], s=1, label=name)
plt.gca().set_box_aspect(1)
plt.xlabel("RA")
plt.ylabel("Dec")
plt.legend()
plt.show()


"""
#### analysis
quick_compare_catalog(lofar, meerkat, lofar_freq, meerkat_freq, "lofar", "meerkat", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, racs,    lofar_freq, racs_freq,    "lofar", "racs", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, tgss,    lofar_freq, tgss_freq,    "lofar", "tgss", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, vlssr,   lofar_freq, vlssr_freq,   "lofar", "vlssr", thres_arc=10, spectral_index_theory=spectral_index_theory)
quick_compare_catalog(racs,  meerkat, racs_freq,  meerkat_freq, "racs",  "meerkat", spectral_index_theory=spectral_index_theory)
"""

#### TODO ####
# add a weighting for point crowding --> downweight sources that are very close to another since they might be confused during matching

#### Parameters
debug = True

#### variables
ras, decs = [], []
correction_factor_global = []
spectral_index_global = []
fitted_flux = []
signal_to_noise = []
fit_to_linear_ratio = []
catalog_weight_factor = []

###################################################
#### catalog three-way combination auto-looper ####
###################################################
for combination in get_triplet_combinations(survey_frequencies, survey_names, required_index=6, skip_index=2):
    local_freqs = (survey_frequencies[combination[0]], survey_frequencies[combination[1]], survey_frequencies[combination[2]])
    local_cats  = (catalogs[combination[0]],           catalogs[combination[1]],           catalogs[combination[2]])
    local_names = (survey_names[combination[0]],       survey_names[combination[1]],       survey_names[combination[2]])
    
    output = compute_flux_correction_factor(local_cats, local_freqs, local_names, debug=debug, return_coord=True)
    
    if output == None:
        print(f"{local_names[0]}, {local_names[1]}, {local_names[2]} did not return any matches")
    else:
        spx, snr, cor, flux, ftl, catw, ra, dec = output

        ras += [ra]; decs += [dec]
        correction_factor_global += [cor]
        spectral_index_global += [spx]
        fitted_flux += [flux]
        signal_to_noise += [snr]
        fit_to_linear_ratio += [ftl]
        catalog_weight_factor += [catw]


###########################################################
#### plotting correction factor for each catalog combo ####
###########################################################
for spx, snr, cor, catw in zip(spectral_index_global, signal_to_noise, correction_factor_global, catalog_weight_factor):
    total_weighting_factor = calculate_weighted_correction_factor(spx, snr, cor, catw)
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
fit_to_linear_ratio = np.concatenate(fit_to_linear_ratio)
catalog_weight_factor = np.concatenate(catalog_weight_factor)


############################################################################
#### plotting correction factor based on all previous catalog matchings ####
############################################################################
total_weighting_factor = calculate_weighted_correction_factor(spectral_index_global, signal_to_noise, correction_factor_global, catalog_weight_factor)
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
# i1, i2, i3, i4 =  match_catalogs_2D(radec_list((lofar, tgss, racs, meerkat)))
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
