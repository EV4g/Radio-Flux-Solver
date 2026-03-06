from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import multiprocessing; multiprocessing.set_start_method('fork') #for windows/mac
from functions import prep_file, get_beam_size, match_catalogs_2D, compute_fluxcal_statistics
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
def get_catalog_matched_flux(cat1, cat2):
    idx_cat1, idx_cat2 = match_catalogs_2D([
        (cat1["ra"], cat1["dec"]),
        (cat2["ra"], cat2["dec"])], thres_arc=2)

    flux1 = cat1['flux_jy'][idx_cat1]
    snr1 = flux1 / cat1['e_flux_jy'][idx_cat1]

    flux2 = cat2['flux_jy'][idx_cat2]
    snr2 = flux2 / cat2['e_flux_jy'][idx_cat2]
    
    return flux1, flux2, snr1, snr2

spectral_index_theory = -0.7
lofar_freq   = 144.6e6 #Hz
racs_freq    = 887.5e6 #Hz
meerkat_freq = 1367e6  #Hz
vlssr_freq   = 73.8e6  #Hz

lofar_files = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0]

# img = bdsf.process_image(
#     lofar_files,
#     thresh_isl=3.0,       # island threshold (sigma)
#     thresh_pix=5.0,       # peak detection threshold (sigma)
#     rms_box=(100, 25),    # (box_size, step_size) for rms map; tune to your image
#     beam=(get_beam_size(lofar_files)),  # (maj_deg, min_deg, PA)
# )

# get calagogs
racs_full    = Table.read(os.getcwd()+"/catalogs/racs/racs_clean.csv")
meerkat_full = Table.read(os.getcwd()+"/catalogs/meerkat/meerkat_clean.csv")
vlssr_full   = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_clean.csv")
lofar        = Table.read(os.getcwd()+"/catalogs/lofar/lofar_sources.fits")

# fix lofar
lofar.rename_column("RA", "ra")
lofar.rename_column("DEC", "dec")
lofar.rename_column('Total_flux', 'flux_jy')
lofar.rename_column('E_Total_flux', 'e_flux_jy')

# check for sources in current file
racs_valid      = sources_in_fits(racs_full['ra'],    racs_full['dec'],    lofar_files)
meerkat_valid   = sources_in_fits(meerkat_full['ra'], meerkat_full['dec'], lofar_files)
vlssr_valid     = sources_in_fits(vlssr_full['ra'],   vlssr_full['dec'],   lofar_files)

# catalog plot
plt.hist(np.log10(racs_full['flux_jy']), alpha=0.6, bins=100, label='racs')
plt.hist(np.log10(meerkat_full['flux_jy']), alpha=0.6, bins=100, label='meerkat')
plt.hist(np.log10(vlssr_full['flux_jy']), alpha=0.6, bins=100, label='vlssr')
plt.hist(np.log10(lofar['Total_flux']), alpha=0.6, bins=50, label='lofar')
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.legend()
plt.show()

plt.hist(np.log10(racs_full['flux_jy'][racs_valid]), alpha=0.6, bins=20, label='racs_v')
plt.hist(np.log10(meerkat_full['flux_jy'][meerkat_valid]), alpha=0.6, bins=20, label='meerkat_v')
plt.hist(np.log10(vlssr_full['flux_jy'][vlssr_valid]), alpha=0.6, bins=20, label='vlssr_v')
plt.hist(np.log10(lofar['Total_flux']), alpha=0.6, bins=50, label='lofar')
plt.legend()
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.show()

# plot catalog as function of position
plt.scatter(racs_full['ra'][racs_valid], racs_full['dec'][racs_valid], s=2)
plt.scatter(meerkat_full['ra'][meerkat_valid], meerkat_full['dec'][meerkat_valid], s=2)
plt.scatter(vlssr_full['ra'][vlssr_valid], vlssr_full['dec'][vlssr_valid], s=2)
plt.scatter(lofar['RA'], lofar['DEC'], s=1)
plt.gca().set_box_aspect(1)
plt.xlabel("RA")
plt.ylabel("Dec")
plt.show()

# remove all non-valid points to reduce syntax clutter
racs    = racs_full[racs_valid]
meerkat = meerkat_full[meerkat_valid]
vlssr   = vlssr_full[vlssr_valid]


#### analysis
# meerkat v lofar
lofar_flux, meerkat_flux, lofar_snr, meerkat_snr = get_catalog_matched_flux(lofar, meerkat)
spectral_flux_ratio, spectral_index_actual, x, log_ratio, scale_factor = compute_fluxcal_statistics(lofar_freq, meerkat_freq, lofar_flux, meerkat_flux)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(meerkat_flux, lofar_flux, s=10, alpha=0.7)
ax[0].plot(x, x * spectral_index_actual, color='purple', ls='--', label="Data fit")
ax[0].plot(x, x, color='black', ls='--', label="x = y")
ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={spectral_index_theory})')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel("Meerkat flux [Jy]"); ax[0].set_ylabel("LOFAR flux [Jy]")
ax[0].legend()

# Log-ratio histogram
ax[1].hist(log_ratio, bins=30, edgecolor='k')
ax[1].axvline(scale_factor, color='red', ls='--', label=f'Median = {scale_factor:.3f}')
ax[1].set_xlabel(r"log$_{10}$(S_LOFAR / S_Meerkat_corrected)")
ax[1].set_ylabel("N")
ax[1].legend()
plt.tight_layout()
#plt.savefig("flux_scale_comparison.png", dpi=150)
plt.show()


# racs v lofar
lofar_flux, racs_flux, lofar_snr, racs_snr = get_catalog_matched_flux(lofar, racs)
spectral_flux_ratio, spectral_index_actual, x, log_ratio, scale_factor = compute_fluxcal_statistics(lofar_freq, racs_freq, lofar_flux, racs_flux)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(racs_flux, lofar_flux, s=10, alpha=0.7)
ax[0].plot(x, x * spectral_index_actual, color='purple', ls='--', label="Data fit")
ax[0].plot(x, x, color='black', ls='--', label="x = y")
ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={spectral_index_theory})')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel("RACS flux [Jy]"); ax[0].set_ylabel("LOFAR flux [Jy]")
ax[0].legend()

# Log-ratio histogram
ax[1].hist(log_ratio, bins=30, edgecolor='k')
ax[1].axvline(scale_factor, color='red', ls='--', label=f'Median = {scale_factor:.3f}')
ax[1].set_xlabel(r"log$_{10}$(S_LOFAR / S_RACS_corrected)")
ax[1].set_ylabel("N")
ax[1].legend()
plt.tight_layout()
#plt.savefig("flux_scale_comparison.png", dpi=150)
plt.show()



# racs v meerkat
racs_flux, meerkat_flux, racs_snr, meerkat_snr = get_catalog_matched_flux(racs, meerkat)
spectral_flux_ratio, spectral_index_actual, x, log_ratio, scale_factor = compute_fluxcal_statistics(meerkat_freq, racs_freq, meerkat_flux, racs_flux)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(racs_flux, meerkat_flux, s=10, alpha=0.7)
ax[0].plot(x, x * spectral_index_actual, color='purple', ls='--', label="Data fit")
ax[0].plot(x, x, color='black', ls='--', label="x = y")
ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={spectral_index_theory})')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel("RACS flux [Jy]"); ax[0].set_ylabel("meerkat flux [Jy]")
ax[0].legend()

# Log-ratio histogram
ax[1].hist(log_ratio, bins=30, edgecolor='k')
ax[1].axvline(scale_factor, color='red', ls='--', label=f'Median = {scale_factor:.3f}')
ax[1].set_xlabel(r"log$_{10}$(S_meerkat / S_RACS_corrected)")
ax[1].set_ylabel("N")
ax[1].legend()
plt.tight_layout()
#plt.savefig("flux_scale_comparison.png", dpi=150)
plt.show()