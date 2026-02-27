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

from functions import spectral_index, get_pixscale, generate_new_wcs, fit_gauss, gaussian_volume

def line(x, a, b): 
    return a * x + b

def log_linspace(mn, mx, n):
    return 10**np.linspace(np.log10(mn), np.log10(mx), n)
    

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

# arrays to fill with values
lofar_flux       = np.zeros_like(cat_ra)
racs_flux        = np.zeros_like(cat_ra)
peak_separation  = np.zeros_like(cat_ra)
sigma_pcov_lofar = np.zeros_like(cat_ra)
sigma_pcov_racs  = np.zeros_like(cat_ra)
local_snr        = np.zeros_like(cat_ra)
local_snr_fit    = np.zeros_like(cat_ra)

# cutout width x height, and pixel and beam scaling
w, h = 1.5 * u.arcmin, 1.5 * u.arcmin
pix_per_deg, nx, ny = get_pixscale(wcs_R, wcs_L, w, h) # pix_per_deg is degrees
beam_area_lofar_px = (np.pi / (4 * np.log(2))) * (lhdul[0].header['BMAJ'] / pix_per_deg.value) * (lhdul[0].header['BMIN'] / pix_per_deg.value)
beam_area_racs_px  = (np.pi / (4 * np.log(2))) * (hdul[0].header['BMAJ'] / pix_per_deg.value) * (hdul[0].header['BMIN'] / pix_per_deg.value)

for i, (ra, dec) in tqdm(enumerate(zip(cat_ra, cat_dec)), total=len(cat_ra)):    
    pos = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")

    # cutout each image around (ra,dec) in its native grid (fast)
    r_co = Cutout2D(racs_data,  position=pos, size=(h, w), wcs=wcs_R, mode="partial", fill_value=np.nan)
    l_co = Cutout2D(lofar_data, position=pos, size=(h, w), wcs=wcs_L, mode="partial", fill_value=np.nan)
    
    # build common output WCS centered on (ra,dec) with your chosen nx,ny,pix_per_deg
    wcs_out = generate_new_wcs(pos, nx, ny, pix_per_deg)
    
    # reproject both cutouts onto the same grid
    racs_cut, racs_fp = reproject_interp((r_co.data, r_co.wcs), wcs_out, shape_out=(nx, ny))
    lofar_cut, lofar_fp = reproject_interp((l_co.data, l_co.wcs), wcs_out, shape_out=(nx, ny))
    
    # gaussian point fit popt=[amplitude, x, y, sigma, z_offset]
    lofar_fit, lofar_popt, lofar_pcov = fit_gauss(lofar_cut, simple=True, debug=True)
    racs_fit,  racs_popt,  racs_pcov  = fit_gauss(racs_cut, simple=True, debug=True)
    
    local_snr[i]     = np.nanmax(lofar_cut) / np.nanstd(lofar_cut) # S/N of actual data
    local_snr_fit[i] = np.nanmax(lofar_fit) / np.nanstd(lofar_cut) # S/N of fitting gaussian relative to data noise
    
    sigma_pcov_lofar[i] = np.sqrt(lofar_pcov[3,3])
    sigma_pcov_racs[i]  = np.sqrt(racs_pcov[3,3])
    
    dist = np.sqrt((lofar_popt[1] - racs_popt[1])**2 + (lofar_popt[2] - racs_popt[2])**2)
    peak_separation[i] = dist
    
    lofar_flux[i] = gaussian_volume(lofar_popt[0], lofar_popt[3]) / beam_area_lofar_px
    racs_flux[i]  = gaussian_volume(racs_popt[0], racs_popt[3]) / beam_area_racs_px
    
    # debug plot
    # is_valid = (lofar_flux[i] > 1e-3) & (racs_flux[i] > 1e-3) & np.isfinite(lofar_flux[i]) & np.isfinite(racs_flux[i]) & (dist < 2) & (sigma_pcov_lofar[i] < 0.3) & (sigma_pcov_racs[i] < 0.3)
    # if is_valid:
    #     fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    #     ax[0, 0].imshow(racs_cut, origin="lower", vmin=-np.nanstd(racs_cut), vmax=5*np.nanstd(racs_cut)); ax[0, 0].set_title("RACS")
    #     ax[1, 0].imshow(lofar_cut, origin="lower", vmin=-np.nanstd(lofar_cut), vmax=5*np.nanstd(lofar_cut)); ax[1, 0].set_title("LOFAR")
    #     ax[0, 1].imshow(racs_fit, origin="lower", vmin=-np.nanstd(racs_cut), vmax=5*np.nanstd(racs_cut))
    #     ax[1, 1].imshow(lofar_fit, origin="lower", vmin=-np.nanstd(lofar_cut), vmax=5*np.nanstd(lofar_cut))
    #     plt.suptitle(f"Source {i} is Valid: {is_valid}")
    #     plt.show()


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
ax[0].scatter(racs_flux[valid], lofar_flux[valid], s=10, alpha=0.7)
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