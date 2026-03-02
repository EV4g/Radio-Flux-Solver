from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from reproject import reproject_interp
import matplotlib.pyplot as plt
import glob
import os
from astropy.nddata.utils import Cutout2D
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from functions import get_pixscale, generate_new_wcs, fit_gauss, gaussian_volume

def line(x, a, b): 
    return a * x + b

def log_linspace(mn, mx, n):
    return 10**np.linspace(np.log10(mn), np.log10(mx), n)
    
def _worker(args):
    ra, dec = args
    return get_flux_fixed(ra, dec)

racs_files = np.sort(glob.glob(os.getcwd()+"/data/racs/*.fits"))
lofar_files = glob.glob(os.getcwd()+"/data/lofar_images/*.fits")
#lofar_files = np.sort(glob.glob("/home/floris/Documents/PhD/Galactic plane/P282+00/ddf/Multi/MSC_deep/*.fits")) #1 restored; 3 dirty
cat = fits.open(os.getcwd()+"/data/P282+00.offset_cat.fits")[1]

# load racs file
racs_file = racs_files[0]
hdul = fits.open(racs_file)
racs_data = hdul[0].data[0, 0]
wcs_R = WCS(hdul[0].header).celestial

# load lofar file
lofar_file = lofar_files[2]
lhdul = fits.open(lofar_file)
lofar_data = lhdul[0].data if len(lhdul[0].data.shape) == 2 else lhdul[0].data[0, 0]
wcs_L = WCS(lhdul[0].header).celestial

# get calagogue ra, dec
cat_ra, cat_dec = cat.data['RA'], cat.data['DEC']

w, h = 1.5 * u.arcmin, 1.5 * u.arcmin

def get_flux(w, h, data1, data2, header1, header2, ra, dec):
    wcs1 = WCS(header1).celestial
    wcs2 = WCS(header2).celestial
    pix_per_deg, nx, ny = get_pixscale(wcs1, wcs2, w, h)

    beam_area_1 = (np.pi / (4*np.log(2))) * (header1['BMAJ'] / pix_per_deg.value) * (header1['BMIN'] / pix_per_deg.value)
    beam_area_2 = (np.pi / (4*np.log(2))) * (header2['BMAJ'] / pix_per_deg.value) * (header2['BMIN'] / pix_per_deg.value)

    pos = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")

    cutout1 = Cutout2D(data1, position=pos, size=(h, w), wcs=wcs1, mode="partial", fill_value=np.nan)
    cutout2 = Cutout2D(data2, position=pos, size=(h, w), wcs=wcs2, mode="partial", fill_value=np.nan)

    wcs_out = generate_new_wcs(pos, nx, ny, pix_per_deg)

    reproj1, _ = reproject_interp((cutout1.data, cutout1.wcs), wcs_out, shape_out=(nx, ny))
    reproj2, _ = reproject_interp((cutout2.data, cutout2.wcs), wcs_out, shape_out=(nx, ny))

    fit1, popt1, pcov1 = fit_gauss(reproj1, simple=True, debug=True)
    fit2, popt2, pcov2 = fit_gauss(reproj2, simple=True, debug=True)

    local_snr     = np.nanmax(reproj1) / np.nanstd(reproj1)
    local_snr_fit = np.nanmax(reproj2) / np.nanstd(reproj2)

    sigma_pcov1 = np.sqrt(pcov1[3, 3])
    sigma_pcov2 = np.sqrt(pcov2[3, 3])

    dist = np.sqrt((popt1[1] - popt2[1])**2 + (popt1[2] - popt2[2])**2)

    flux1 = gaussian_volume(popt1[0], popt1[3]) / beam_area_1  # ← fixed: was popt2[0]
    flux2 = gaussian_volume(popt2[0], popt2[3]) / beam_area_2

    return flux1, flux2, dist, sigma_pcov1, sigma_pcov2, local_snr, local_snr_fit

get_flux_fixed = partial(get_flux, w, h, lofar_data, racs_data, lhdul[0].header, hdul[0].header)

with ProcessPoolExecutor(max_workers=24) as executor:
    results = list(tqdm(executor.map(_worker, zip(cat_ra, cat_dec), chunksize=8), total=len(cat_ra)))

lofar_flux, racs_flux, peak_separation, sigma_pcov_lofar, sigma_pcov_racs, local_snr, local_snr_fit = map(np.array, zip(*results))

# quality filtering
valid = (lofar_flux > 1e-3) & (racs_flux > 1e-3) & np.isfinite(lofar_flux) & np.isfinite(racs_flux) \
        & (peak_separation < 2) & (sigma_pcov_lofar < 0.3) & (sigma_pcov_racs < 0.3) \
        & (local_snr > 5) & (local_snr_fit > 4)

#### analysis
spectral_index_alpha = -0.7
lofar_freq = 144.6e6 #Hz
racs_freq = 887.5e6 #Hz
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