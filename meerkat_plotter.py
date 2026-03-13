import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from reproject import reproject_interp
import matplotlib.pyplot as plt
import glob

from functions import spectral_index, get_survey_file, cutout_to_galactic_wh, get_pixscale, generate_new_wcs

warnings.filterwarnings("ignore", category=FITSFixedWarning)
basedir = "/home/floris/Documents/PhD/Galactic plane/"

meerkat_files = np.sort(glob.glob(basedir+"LOFAR_and_MeerKAT/data/meerkat/*.fits"))
#lofar_files = np.sort(glob.glob(basedir+"P282+00/ddf/Multi/HogbomMulti/*.fits")) #1 restored; 3 dirty
#lofar_files = np.sort(glob.glob(basedir+"P282+00/ddf/Multi/MSC_deep/*.fits")) #1 restored; 3 dirty
lofar_files = np.sort(glob.glob(basedir+"LOFAR_and_MeerKAT/data/lofar/*.fits")) #0 high res; 1 low res

# cutout (galactic lat,lon)
cutout_lat = [-0.72, -0.08]
cutout_lon = [34.4, 34.92]
center, width, height = cutout_to_galactic_wh(cutout_lon, cutout_lat)

# load meerkat data
meerkat_file = get_survey_file(meerkat_files, center)
mhdul = fits.open(meerkat_file)
meerkat_data = mhdul[0].data[0, 0]
wcs_M = WCS(mhdul[0].header).celestial

# load lofar file
lofar_file = lofar_files[0]
lhdul = fits.open(lofar_file)
lofar_data = lhdul[0].data if len(lhdul[0].data.shape) == 2 else lhdul[0].data[0, 0]
wcs_L = WCS(lhdul[0].header).celestial

# get pixel scale and image size, and build new wcs
pixscale, nx, ny = get_pixscale(wcs_M, wcs_L, width, height)
wcs_out = generate_new_wcs(center, nx, ny, pixscale)

# reproject both to the same grid
meerkat_cut, meerkat_fp = reproject_interp((meerkat_data, wcs_M), wcs_out, shape_out=(ny, nx))
lofar_cut,   lofar_fp   = reproject_interp((lofar_data,   wcs_L), wcs_out, shape_out=(ny, nx))

#### analysis
plt.imshow(meerkat_cut, vmin=-np.nanstd(meerkat_cut), vmax=5*np.nanstd(meerkat_cut), origin='lower')
plt.colorbar()
plt.show()

plt.imshow(lofar_cut, vmin=-np.nanstd(lofar_cut), vmax=5*np.nanstd(lofar_cut), origin='lower')
plt.colorbar()
plt.show()

# spectral index map
spx = spectral_index(meerkat_cut, lofar_cut*3.86, 1359.7, 144.6)
plt.imshow(spx, origin='lower', vmin=-1.2, vmax=0.2, cmap='coolwarm')
plt.ylabel("y (pixels)")
plt.xlabel("x (pixels)")
plt.colorbar(label=r"spectral index $\alpha$")
plt.title("SNR source spectral index")
plt.show()
