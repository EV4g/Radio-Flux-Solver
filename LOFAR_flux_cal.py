from astropy.io import fits
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import multiprocessing; multiprocessing.set_start_method('fork') #for windows/mac
from functions import get_flux_batch, prep_file

def line(x, a, b): 
    return a * x + b

def log_linspace(mn, mx, n):
    return 10**np.linspace(np.log10(mn), np.log10(mx), n)

spectral_index_alpha = -0.7
lofar_freq = 144.6e6 #Hz
racs_freq = 887.5e6 #Hz
meerkat_freq = 1367e6 #Hz

racs_files = np.sort(glob.glob(os.getcwd()+"/data/racs/*.fits"))
lofar_files = glob.glob(os.getcwd()+"/data/lofar/*.fits")
meerkat_files = np.sort(glob.glob(os.getcwd()+"/data/meerkat/*.fits"))
#lofar_files = np.sort(glob.glob("/home/floris/Documents/PhD/Galactic plane/P282+00/ddf/Multi/MSC_deep/*.fits")) #1 restored; 3 dirty
cat = fits.open(os.getcwd()+"/data/P282+00.offset_cat.fits")[1]

# load lofar and racs files
racs_data, racs_header, wcs_R = prep_file(racs_files[0])
lofar_data, lofar_header, wcs_L = prep_file(lofar_files[2])
meerkat_data, meerkat_header, wcs_M = prep_file(meerkat_files[11])

# get calagogue ra, dec
cat_ra, cat_dec = cat.data['RA'], cat.data['DEC']

w, h = 1.5 * u.arcmin, 1.5 * u.arcmin

lofar_flux, racs_flux, peak_separation, sigma_pcov_lofar, sigma_pcov_racs, local_snr, local_snr_fit = get_flux_batch(w, h, lofar_data, racs_data, lofar_header, racs_header, cat_ra, cat_dec)

# quality filtering
valid = (lofar_flux > 1e-3) & (racs_flux > 1e-3) & np.isfinite(lofar_flux) & np.isfinite(racs_flux) \
        & (peak_separation < 2) & (sigma_pcov_lofar < 0.3) & (sigma_pcov_racs < 0.3) \
        & (local_snr > 5) & (local_snr_fit > 4)

#### analysis
spectral_flux_ratio = (lofar_freq / racs_freq)**spectral_index_alpha

# fitting a line through valid points
index = 10**np.mean(np.log10(lofar_flux[valid] / racs_flux[valid]))
x = log_linspace(np.min(racs_flux[valid]), np.max(racs_flux[valid]), 10)

# calculate ratio
ratio = lofar_flux[valid] / (racs_flux[valid] * spectral_flux_ratio)
valid_ratio = np.isfinite(ratio) & (ratio > 0)
log_ratio = np.log10(ratio)

scale_factor = np.median(log_ratio)
scatter      = np.std(log_ratio)
N            = len(log_ratio)
stderr       = scatter / np.sqrt(N)

print(f"N valid sources      : {N}")
print(f"Median log10(ratio)  : {scale_factor:.4f}  ({10**scale_factor:.4f}×)")
print(f"Scatter (1σ)         : {scatter:.4f} dex")
print(f"Uncertainty on median: ±{stderr:.4f} dex")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(racs_flux[valid], lofar_flux[valid], s=10, alpha=0.7, c=local_snr[valid])
ax[0].plot(x, x * index, color='purple', ls='--', label="Data fit")
ax[0].plot(x, x, color='black', ls='--', label="x = y")
ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={spectral_index_alpha})')
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


#### testing

lofar_flux, meerkat_flux, peak_separation, sigma_pcov_lofar, sigma_pcov_meerkat, local_snr, local_snr_fit = get_flux_batch(w, h, lofar_data, meerkat_data, lofar_header, meerkat_header, cat_ra, cat_dec)

# quality filtering
valid = (lofar_flux > 1e-3) & (meerkat_flux > 5e-3) & np.isfinite(lofar_flux) & np.isfinite(meerkat_flux) \
        & (peak_separation < 2) & (sigma_pcov_lofar < 0.3) & (sigma_pcov_meerkat < 0.3) \
        & (local_snr > 5) & (local_snr_fit > 4)

#### analysis
spectral_flux_ratio = (lofar_freq / meerkat_freq)**spectral_index_alpha

# fitting a line through valid point
index = 10**np.mean(np.log10(lofar_flux[valid] / meerkat_flux[valid]))
x = log_linspace(np.min(meerkat_flux[valid]), np.max(meerkat_flux[valid]), 10)

# calculate ratio
ratio = lofar_flux[valid] / (meerkat_flux[valid] * spectral_flux_ratio)
valid_ratio = np.isfinite(ratio) & (ratio > 0)
log_ratio = np.log10(ratio)

scale_factor = np.median(log_ratio)
scatter      = np.std(log_ratio)
N            = len(log_ratio)
stderr       = scatter / np.sqrt(N)

print(f"N valid sources      : {N}")
print(f"Median log10(ratio)  : {scale_factor:.4f}  ({10**scale_factor:.4f}×)")
print(f"Scatter (1σ)         : {scatter:.4f} dex")
print(f"Uncertainty on median: ±{stderr:.4f} dex")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(meerkat_flux[valid], lofar_flux[valid], s=10, alpha=0.7, c=local_snr[valid])
ax[0].plot(x, x * index, color='purple', ls='--', label="Data fit")
ax[0].plot(x, x, color='black', ls='--', label="x = y")
ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={spectral_index_alpha})')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel("MeerKat flux [Jy]"); ax[0].set_ylabel("LOFAR flux [Jy]")
ax[0].legend()

# Log-ratio histogram
ax[1].hist(log_ratio, bins=15, edgecolor='k')
ax[1].axvline(scale_factor, color='red', ls='--', label=f'Median = {scale_factor:.3f}')
ax[1].set_xlabel(r"log$_{10}$(S_LOFAR / S_MeerKat_corrected)")
ax[1].set_ylabel("N")
ax[1].legend()
plt.tight_layout()

#plt.savefig("flux_scale_comparison.png", dpi=150)
plt.show()