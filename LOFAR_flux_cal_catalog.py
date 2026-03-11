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
lofar_freq   = 144.6e6 #Hz
racs_freq    = 887.5e6 #Hz
meerkat_freq = 1367e6  #Hz
vlssr_freq   = 73.8e6  #Hz
tgss_freq    = 150e6   #Hz

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
racs_full    = Table.read(os.getcwd()+"/catalogs/racs/racs_clean.csv")
meerkat_full = Table.read(os.getcwd()+"/catalogs/meerkat/meerkat_clean.csv")
vlssr_full   = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_clean.csv")
tgss_full    = Table.read(os.getcwd()+"/catalogs/tgss/tgss_clean.fits")
lofar        = Table.read(os.getcwd()+"/catalogs/lofar/lofar_sources_pipeline.fits")

# fix lofar
lofar.rename_column("RA", "ra")
lofar.rename_column("DEC", "dec")
lofar.rename_column('Total_flux', 'flux_jy')
lofar.rename_column('E_Total_flux', 'e_flux_jy')

# check for sources in current file
racs_valid      = sources_in_fits(racs_full['ra'],    racs_full['dec'],    lofar_files)
meerkat_valid   = sources_in_fits(meerkat_full['ra'], meerkat_full['dec'], lofar_files)
vlssr_valid     = sources_in_fits(vlssr_full['ra'],   vlssr_full['dec'],   lofar_files)
tgss_valid      = sources_in_fits(tgss_full['ra'],    tgss_full['dec'],    lofar_files)

# remove all non-valid points to reduce syntax clutter
racs    = racs_full[racs_valid]
meerkat = meerkat_full[meerkat_valid]
vlssr   = vlssr_full[vlssr_valid]
tgss    = tgss_full[tgss_valid]

# catalog plot
plt.hist(np.log10(racs_full['flux_jy']),    alpha=0.6, bins=100, label='racs')
plt.hist(np.log10(meerkat_full['flux_jy']), alpha=0.6, bins=100, label='meerkat')
plt.hist(np.log10(vlssr_full['flux_jy']),   alpha=0.6, bins=100, label='vlssr')
plt.hist(np.log10(tgss_full['flux_jy']),    alpha=0.6, bins=100, label='tgss')
plt.hist(np.log10(lofar['flux_jy']),        alpha=0.6, bins=50,  label='lofar')
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.legend()
plt.show()

plt.hist(np.log10(racs['flux_jy']),    alpha=0.6, bins=20, label='racs_v')
plt.hist(np.log10(meerkat['flux_jy']), alpha=0.6, bins=20, label='meerkat_v')
plt.hist(np.log10(vlssr['flux_jy']),   alpha=0.6, bins=20, label='vlssr_v')
plt.hist(np.log10(tgss['flux_jy']),    alpha=0.6, bins=100, label='tgss_v')
plt.hist(np.log10(lofar['flux_jy']),   alpha=0.6, bins=50, label='lofar')
plt.legend()
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.show()

# plot catalog as function of position
plt.scatter(racs['ra'],    racs['dec'],    s=2, label='racs')
plt.scatter(meerkat['ra'], meerkat['dec'], s=2, label='meerkat')
plt.scatter(vlssr['ra'],   vlssr['dec'],   s=2, label='vlssr')
plt.scatter(tgss['ra'],    tgss['dec'],    s=2, label='tgss')
plt.scatter(lofar['ra'],   lofar['dec'],   s=1, label='lofar')
plt.gca().set_box_aspect(1)
plt.xlabel("RA")
plt.ylabel("Dec")
plt.legend()
plt.show()



#### analysis
quick_compare_catalog(lofar, meerkat, lofar_freq, meerkat_freq, "lofar", "meerkat", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, racs,    lofar_freq, racs_freq,    "lofar", "racs", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, tgss,    lofar_freq, tgss_freq,    "lofar", "tgss", spectral_index_theory=spectral_index_theory)
quick_compare_catalog(lofar, vlssr,   lofar_freq, vlssr_freq,   "lofar", "vlssr", thres_arc=10, spectral_index_theory=spectral_index_theory)
quick_compare_catalog(racs,  meerkat, racs_freq,  meerkat_freq, "racs",  "meerkat", spectral_index_theory=spectral_index_theory)

"""
#### beam dependant flux analysis
i1, i2 = match_catalogs_2D(radec_list((tgss, lofar)), thres_arc=1.5)
plot_location_dependant_index(lofar['ra'][i2], lofar['dec'][i2], lofar['flux_jy'][i2] / tgss['flux_jy'][i1])

i1, i2 = match_catalogs_2D(radec_list((racs, lofar)), thres_arc=1.5)
plot_location_dependant_index(lofar['ra'][i2], lofar['dec'][i2], lofar['flux_jy'][i2] / racs['flux_jy'][i1])

i1, i2 = match_catalogs_2D(radec_list((meerkat, lofar)), thres_arc=1.5)
plot_location_dependant_index(lofar['ra'][i2], lofar['dec'][i2], lofar['flux_jy'][i2] / meerkat['flux_jy'][i1])

# combined plot
i1, i2 = match_catalogs_2D(radec_list((tgss, lofar)), thres_arc=1.5)
f = lofar['flux_jy'][i2] / tgss['flux_jy'][i1]
plt.scatter(lofar['ra'][i2], lofar['dec'][i2], c=f / np.max(f))

i1, i2 = match_catalogs_2D(radec_list((racs, lofar)), thres_arc=1.5)
f = lofar['flux_jy'][i2] / racs['flux_jy'][i1]
plt.scatter(lofar['ra'][i2], lofar['dec'][i2], c=f/np.max(f))

i1, i2 = match_catalogs_2D(radec_list((meerkat, lofar)), thres_arc=1.5)
f = lofar['flux_jy'][i2] / meerkat['flux_jy'][i1]
plt.scatter(lofar['ra'][i2], lofar['dec'][i2], c=f/np.max(f))


plt.colorbar()
plt.gca().set_box_aspect(1)
plt.show()
"""

#### spectral index
# combinations that are useful to consider:
# (lofar) + racs + meerkat
# (lofar) + tgss + meerkat
# (lofar) + tgss + racs
# vlssr doesn't have proper matches with the other surveys


i1, i2, i3 =  match_catalogs_2D(radec_list((vlssr, lofar, racs)))





