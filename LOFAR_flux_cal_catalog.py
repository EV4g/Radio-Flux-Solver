from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import multiprocessing; multiprocessing.set_start_method('fork') #for windows/mac
from functions import prep_file, get_beam_size, match_catalogs_2D, compute_fluxcal_statistics, get_spectral_index
from astropy.table import Table
import bdsf

"""Plot (log)ratio as function of position in field."""
def plot_location_dependant_index(ra, dec, ratio):
    plt.scatter(ra, dec, c=ratio)
    plt.colorbar()
    plt.gca().set_box_aspect(1)
    plt.show()

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

"""w.i.p."""
def merge_catalogs(cats, thres_arc=2):
    indices = match_catalogs_2D(radec_list(cats), thres_arc=thres_arc)
    
    catalog = Table()
    catalog['ra'] = cats[0]['ra'][indices[0]]
    catalog['dec'] = cats[0]['dec'][indices[0]]
    
    return catalog
    



def radec_list(cats):
    radec_list = []
    for cat in cats:
        radec_list.append((cat['ra'], cat['dec']))
    return radec_list        

spectral_index_theory = -0.7
lofar_freq     = 144.6e6 #Hz
racs_freq      = 887.5e6 #Hz
meerkat_freq   = 1359.7e6  #Hz
vlssr_freq     = 73.8e6  #Hz
tgss_freq      = 150e6   #Hz
gleam_300_freq = 300e6 #Hz

lofar_files = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0]

# img = bdsf.process_image(
#     lofar_files,
#     thresh_isl=3.0,       # island threshold (sigma)
#     thresh_pix=5.0,       # peak detection threshold (sigma)
#     rms_box=(100, 25),    # (box_size, step_size) for rms map; tune to your image
#     beam=(get_beam_size(lofar_files)),  # (maj_deg, min_deg, PA)
# )
# img.write_catalog(outfile="lofar_sources_pipeline.fits", format="fits", catalog_type="srl", clobber=True)

# get calagogs
racs_full      = Table.read(os.getcwd()+"/catalogs/racs/racs_clean.csv")
meerkat_full   = Table.read(os.getcwd()+"/catalogs/meerkat/meerkat_clean.csv")
vlssr_full     = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_clean.csv")
tgss_full      = Table.read(os.getcwd()+"/catalogs/tgss/tgss_clean.fits")
gleam_300_full = Table.read(os.getcwd()+"/catalogs/gleam/gleam_300_clean.fits")
lofar          = Table.read(os.getcwd()+"/catalogs/lofar/lofar_sources_pipeline.fits")

# fix lofar
lofar.rename_column("RA", "ra")
lofar.rename_column("DEC", "dec")
lofar.rename_column('Total_flux', 'flux_jy')
lofar.rename_column('E_Total_flux', 'e_flux_jy')

# check for sources in current file
racs_valid      = sources_in_fits(racs_full['ra'],      racs_full['dec'],       lofar_files)
meerkat_valid   = sources_in_fits(meerkat_full['ra'],   meerkat_full['dec'],    lofar_files)
vlssr_valid     = sources_in_fits(vlssr_full['ra'],     vlssr_full['dec'],      lofar_files)
tgss_valid      = sources_in_fits(tgss_full['ra'],      tgss_full['dec'],       lofar_files)
gleam_300_valid = sources_in_fits(gleam_300_full['ra'], gleam_300_full['dec'],  lofar_files)

# remove all non-valid points to reduce syntax clutter
racs      = racs_full[racs_valid]
meerkat   = meerkat_full[meerkat_valid]
vlssr     = vlssr_full[vlssr_valid]
tgss      = tgss_full[tgss_valid]
gleam_300 = gleam_300_full[gleam_300_valid]

"""
# catalog plot
plt.hist(np.log10(racs_full['flux_jy']),      alpha=0.6, bins=100, label='racs')
plt.hist(np.log10(meerkat_full['flux_jy']),   alpha=0.6, bins=100, label='meerkat')
plt.hist(np.log10(vlssr_full['flux_jy']),     alpha=0.6, bins=100, label='vlssr')
plt.hist(np.log10(tgss_full['flux_jy']),      alpha=0.6, bins=100, label='tgss')
plt.hist(np.log10(gleam_300_full['flux_jy']), alpha=0.6, bins=100, label='gleam')
plt.hist(np.log10(lofar['flux_jy']),          alpha=0.6, bins=50,  label='lofar')
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.legend()
plt.show()

plt.hist(np.log10(racs['flux_jy']),      alpha=0.6, bins=20, label='racs_v')
plt.hist(np.log10(meerkat['flux_jy']),   alpha=0.6, bins=20, label='meerkat_v')
plt.hist(np.log10(vlssr['flux_jy']),     alpha=0.6, bins=20, label='vlssr_v')
plt.hist(np.log10(tgss['flux_jy']),      alpha=0.6, bins=50, label='tgss_v')
plt.hist(np.log10(gleam_300['flux_jy']), alpha=0.6, bins=30, label='gleam_v')
plt.hist(np.log10(lofar['flux_jy']),     alpha=0.6, bins=40, label='lofar')
plt.legend()
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.show()

# plot catalog as function of position
plt.scatter(racs['ra'],      racs['dec'],      s=2, label='racs')
plt.scatter(meerkat['ra'],   meerkat['dec'],   s=2, label='meerkat')
plt.scatter(vlssr['ra'],     vlssr['dec'],     s=2, label='vlssr')
plt.scatter(tgss['ra'],      tgss['dec'],      s=2, label='tgss')
plt.scatter(gleam_300['ra'], gleam_300['dec'], s=2, label='gleam')
plt.scatter(lofar['ra'],     lofar['dec'],     s=1, label='lofar')
plt.gca().set_box_aspect(1)
plt.xlabel("RA")
plt.ylabel("Dec")
plt.legend()
plt.show()
"""

"""
#### analysis
quick_compare_catalog(lofar, meerkat, lofar_freq, meerkat_freq, "lofar", "meerkat", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, racs,    lofar_freq, racs_freq,    "lofar", "racs", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, tgss,    lofar_freq, tgss_freq,    "lofar", "tgss", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, vlssr,   lofar_freq, vlssr_freq,   "lofar", "vlssr", thres_arc=10, spectral_index_theory=spectral_index_theory)
quick_compare_catalog(racs,  meerkat, racs_freq,  meerkat_freq, "racs",  "meerkat", spectral_index_theory=spectral_index_theory)
"""

#### spectral index
# combinations that are useful to consider:
# (lofar) + racs + meerkat
# (lofar) + tgss + meerkat
# (lofar) + tgss + racs
# (lofar) + tgss + racs + meerkat
# vlssr doesn't have proper matches with the other surveys

def get_flux_from_index(spectral_index, reference_flux, current_frequency, reference_frequency):
    return reference_flux * (current_frequency / reference_frequency) ** spectral_index

# arrays to keep track of all individual runs
ras, decs = [], []
correction_factor_global = []
spectral_index_global = []

##########################
# lofar + racs + meerkat #
##########################
i1, i2, i3 =  match_catalogs_2D(radec_list((lofar, racs, meerkat)))
freqs = np.array([lofar_freq, racs_freq, meerkat_freq]) * 1e-6
spectral_indices = []             # fitted spectral index based on racs, meerkat
extrapolated_flux_linear = []     # extrapolated flux when assuming a simple -0.7
extrapolated_flux_fit = []        # extrapolated flux when fitting the fluxes from racs and meerkat
lofar_uncorrected_flux = []       # current lofar flux, no corrections
lofar_uncorrected_flux_error = [] # current lofar flux error, no corrections

for i, (lofar_i, racs_i, meerkat_i) in enumerate(zip(i1, i2, i3)):
    lofar_F, racs_F, meerkat_F = lofar['flux_jy'][lofar_i], racs['flux_jy'][racs_i], meerkat['flux_jy'][meerkat_i]
    spectral_index = get_spectral_index(racs_F, meerkat_F, freqs[1], freqs[2])
    spectral_indices.append(spectral_index)
    
    lofar_uncorrected_flux.append(lofar_F)
    lofar_uncorrected_flux_error.append(lofar['e_flux_jy'][lofar_i])
    
    # lofar flux when using racs flux and extrapolating back to lofar freq
    extrapolated_flux_fit.append(get_flux_from_index(spectral_index, racs_F, freqs[0], freqs[1]))
    extrapolated_flux_linear.append(0.5 * (get_flux_from_index(-0.7, racs_F, freqs[0], freqs[1]) + get_flux_from_index(-0.7, meerkat_F, freqs[0], freqs[2])))
    
    
    plt.plot(freqs, [lofar_F, racs_F, meerkat_F])
    
plt.yscale('log')
plt.ylabel(r"log$_{10}$(Jy)")
plt.xlabel("Frequency (MHz)")
plt.show()

extrapolated_flux_fit = np.array(extrapolated_flux_fit)
lofar_uncorrected_flux = np.array(lofar_uncorrected_flux)
lofar_uncorrected_flux_error = np.array(lofar_uncorrected_flux_error)
spectral_indices = np.array(spectral_indices)

valid_factor = (spectral_indices > -1) & (spectral_indices < 0)
correction_factor = extrapolated_flux_fit / lofar_uncorrected_flux


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
plt.xlabel("Lofar linear flux (Jy)")
plt.ylabel("Lofar fitted flux (Jy)")
plt.title(r"Lofar flux, $\alpha$=-0.7 vs fitted")
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

# signal to noise
snr = lofar_uncorrected_flux / lofar_uncorrected_flux_error

# distance of spectral index to alpha = -0.7
spectral_difference = np.abs(spectral_indices + 0.7)
spectral_difference_factor = (1 - (spectral_difference / np.max(spectral_difference)))**2

# logarithmic weighted mean of the flux correction factor
weighted_mean_correction = 10**(np.mean(snr * spectral_difference_factor * np.log10(correction_factor)) / np.mean(snr * spectral_difference_factor))

# add arrays to global ones to keep track
ras += [lofar['ra'][i1]]
decs += [lofar['dec'][i1]]
correction_factor_global += [correction_factor]
spectral_index_global += [spectral_indices]



##########################
# lofar + tgss + meerkat #
##########################
i1, i2, i3 =  match_catalogs_2D(radec_list((lofar, tgss, meerkat)))
freqs = np.array([lofar_freq, tgss_freq, meerkat_freq]) * 1e-6
spectral_indices = []             # fitted spectral index based on racs, meerkat
extrapolated_flux_linear = []     # extrapolated flux when assuming a simple -0.7
extrapolated_flux_fit = []        # extrapolated flux when fitting the fluxes from racs and meerkat
lofar_uncorrected_flux = []       # current lofar flux, no corrections
lofar_uncorrected_flux_error = [] # current lofar flux error, no corrections

for i, (lofar_i, tgss_i, meerkat_i) in enumerate(zip(i1, i2, i3)):
    lofar_F, tgss_F, meerkat_F = lofar['flux_jy'][lofar_i], tgss['flux_jy'][tgss_i], meerkat['flux_jy'][meerkat_i]
    spectral_index = get_spectral_index(tgss_F, meerkat_F, freqs[1], freqs[2])
    spectral_indices.append(spectral_index)
    
    lofar_uncorrected_flux.append(lofar_F)
    lofar_uncorrected_flux_error.append(lofar['e_flux_jy'][lofar_i])
    
    # lofar flux when using racs flux and extrapolating back to lofar freq
    extrapolated_flux_fit.append(get_flux_from_index(spectral_index, tgss_F, freqs[0], freqs[1]))
    extrapolated_flux_linear.append(0.5 * (get_flux_from_index(-0.7, tgss_F, freqs[0], freqs[1]) + get_flux_from_index(-0.7, meerkat_F, freqs[0], freqs[2])))
    
    
    plt.plot(freqs, [lofar_F, tgss_F, meerkat_F])
    
plt.yscale('log')
plt.ylabel(r"log$_{10}$(Jy)")
plt.xlabel("Frequency (MHz)")
plt.show()

extrapolated_flux_fit = np.array(extrapolated_flux_fit)
lofar_uncorrected_flux = np.array(lofar_uncorrected_flux)
lofar_uncorrected_flux_error = np.array(lofar_uncorrected_flux_error)
spectral_indices = np.array(spectral_indices)

valid_factor = (spectral_indices > -1) & (spectral_indices < 0)
correction_factor = extrapolated_flux_fit / lofar_uncorrected_flux


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
plt.xlabel("Lofar linear flux (Jy)")
plt.ylabel("Lofar fitted flux (Jy)")
plt.title(r"Lofar flux, $\alpha$=-0.7 vs fitted")
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

# signal to noise
snr = lofar_uncorrected_flux / lofar_uncorrected_flux_error

# distance of spectral index to alpha = -0.7
spectral_difference = np.abs(spectral_indices + 0.7)
spectral_difference_factor = (1 - (spectral_difference / np.max(spectral_difference)))**2

# logarithmic weighted mean of the flux correction factor
weighted_mean_correction = 10**(np.mean(snr * spectral_difference_factor * np.log10(correction_factor)) / np.mean(snr * spectral_difference_factor))

# add arrays to global ones to keep track
ras += [lofar['ra'][i1]]
decs += [lofar['dec'][i1]]
correction_factor_global += [correction_factor]
spectral_index_global += [spectral_indices]



#######################
# lofar + tgss + racs #
#######################
i1, i2, i3 =  match_catalogs_2D(radec_list((lofar, tgss, racs)))
freqs = np.array([lofar_freq, tgss_freq, racs_freq]) * 1e-6
spectral_indices = []             # fitted spectral index based on racs, meerkat
extrapolated_flux_linear = []     # extrapolated flux when assuming a simple -0.7
extrapolated_flux_fit = []        # extrapolated flux when fitting the fluxes from racs and meerkat
lofar_uncorrected_flux = []       # current lofar flux, no corrections
lofar_uncorrected_flux_error = [] # current lofar flux error, no corrections

for i, (lofar_i, tgss_i, racs_i) in enumerate(zip(i1, i2, i3)):
    lofar_F, tgss_F, racs_F = lofar['flux_jy'][lofar_i], tgss['flux_jy'][tgss_i], racs['flux_jy'][racs_i]
    spectral_index = get_spectral_index(tgss_F, racs_F, freqs[1], freqs[2])
    spectral_indices.append(spectral_index)
    
    lofar_uncorrected_flux.append(lofar_F)
    lofar_uncorrected_flux_error.append(lofar['e_flux_jy'][lofar_i])
    
    # lofar flux when using racs flux and extrapolating back to lofar freq
    extrapolated_flux_fit.append(get_flux_from_index(spectral_index, tgss_F, freqs[0], freqs[1]))
    extrapolated_flux_linear.append(0.5 * (get_flux_from_index(-0.7, tgss_F, freqs[0], freqs[1]) + get_flux_from_index(-0.7, racs_F, freqs[0], freqs[2])))
    
    
    plt.plot(freqs, [lofar_F, tgss_F, racs_F])
    
plt.yscale('log')
plt.ylabel(r"log$_{10}$(Jy)")
plt.xlabel("Frequency (MHz)")
plt.show()

extrapolated_flux_fit = np.array(extrapolated_flux_fit)
lofar_uncorrected_flux = np.array(lofar_uncorrected_flux)
lofar_uncorrected_flux_error = np.array(lofar_uncorrected_flux_error)
spectral_indices = np.array(spectral_indices)

valid_factor = (spectral_indices > -1) & (spectral_indices < 0)
correction_factor = extrapolated_flux_fit / lofar_uncorrected_flux


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
plt.xlabel("Lofar linear flux (Jy)")
plt.ylabel("Lofar fitted flux (Jy)")
plt.title(r"Lofar flux, $\alpha$=-0.7 vs fitted")
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

# signal to noise
snr = lofar_uncorrected_flux / lofar_uncorrected_flux_error

# distance of spectral index to alpha = -0.7
spectral_difference = np.abs(spectral_indices + 0.7)
spectral_difference_factor = (1 - (spectral_difference / np.max(spectral_difference)))**2

# logarithmic weighted mean of the flux correction factor
weighted_mean_correction = 10**(np.mean(snr * spectral_difference_factor * np.log10(correction_factor)) / np.mean(snr * spectral_difference_factor))

# add arrays to global ones to keep track
ras += [lofar['ra'][i1]]
decs += [lofar['dec'][i1]]
correction_factor_global += [correction_factor]
spectral_index_global += [spectral_indices]


#################################
# lofar + tgss + racs + meerkat #
#################################
i1, i2, i3, i4 =  match_catalogs_2D(radec_list((lofar, tgss, racs, meerkat)))
freqs = np.array([lofar_freq, tgss_freq, racs_freq, meerkat_freq]) * 1e-6
spectral_indices = []             # fitted spectral index based on racs, meerkat
extrapolated_flux_linear = []     # extrapolated flux when assuming a simple -0.7
extrapolated_flux_fit = []        # extrapolated flux when fitting the fluxes from racs and meerkat
lofar_uncorrected_flux = []       # current lofar flux, no corrections
lofar_uncorrected_flux_error = [] # current lofar flux error, no corrections

for i, (lofar_i, tgss_i, racs_i, meerkat_i) in enumerate(zip(i1, i2, i3, i4)):
    lofar_F, tgss_F, racs_F, meerkat_F = lofar['flux_jy'][lofar_i], tgss['flux_jy'][tgss_i], racs['flux_jy'][racs_i], meerkat['flux_jy'][meerkat_i]
    #spectral_index = get_spectral_index(tgss_F, racs_F, freqs[1], freqs[2])
    #spectral_indices.append(spectral_index)
    
    lofar_uncorrected_flux.append(lofar_F)
    lofar_uncorrected_flux_error.append(lofar['e_flux_jy'][lofar_i])
    
    # lofar flux when using racs flux and extrapolating back to lofar freq
    #extrapolated_flux_fit.append(get_flux_from_index(spectral_index, tgss_F, freqs[0], freqs[1]))
    #extrapolated_flux_linear.append(0.5 * (get_flux_from_index(-0.7, tgss_F, freqs[0], freqs[1]) + get_flux_from_index(-0.7, racs_F, freqs[0], freqs[2])))
    
    
    plt.plot(freqs, [lofar_F, tgss_F, racs_F, meerkat_F])
    
plt.yscale('log')
plt.ylabel(r"log$_{10}$(Jy)")
plt.xlabel("Frequency (MHz)")
plt.show()

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
