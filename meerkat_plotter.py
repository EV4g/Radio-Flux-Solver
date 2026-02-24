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

from functions import spectral_index, get_meerkat_file, cutout_to_galactic_wh, get_pixscale, generate_new_wcs

warnings.filterwarnings("ignore", category=FITSFixedWarning)
basedir = "/home/floris/Documents/PhD/Galactic plane/"

meerkat_files = np.sort(glob.glob(basedir+"LOFAR_and_MeerKAT/data/smgps_mfs_images/*.fits"))
#lofar_files = np.sort(glob.glob(basedir+"P282+00/ddf/Multi/HogbomMulti/*.fits")) #1 restored; 3 dirty
lofar_files = np.sort(glob.glob(basedir+"P282+00/ddf/Multi/MSC_deep/*.fits")) #1 restored; 3 dirty
#lofar_files = np.sort(glob.glob(basedir+"LOFAR_and_MeerKAT/data/lofar_images/*.fits")) #0 high res; 1 low res

# cutout (galactic lat,lon)
cutout_lat = [-0.75, -0.05]
cutout_lon = [34.25, 34.95]
center, width, height = cutout_to_galactic_wh(cutout_lon, cutout_lat)

# load meerkat data
meerkat_file = get_meerkat_file(meerkat_files, center)
mhdul = fits.open(meerkat_file)
meerkat_data = mhdul[0].data[0, 0]
wcs_M = WCS(mhdul[0].header).celestial

# load lofar file
lofar_file = lofar_files[1]
lhdul = fits.open(lofar_file)
lofar_data = lhdul[0].data if len(lhdul[0].data.shape) == 2 else lhdul[0].data[0, 0]
wcs_L = WCS(lhdul[0].header).celestial

# get pixel scale and image size, and build new wcs
pixscale, nx, ny = get_pixscale(wcs_M, wcs_L, width, height)
wcs_out = generate_new_wcs(center, nx, ny, pixscale)

# reproject both to the same grid
meerkat_cut, meerkat_fp = reproject_interp((meerkat_data, wcs_M), wcs_out, shape_out=(nx, ny))
lofar_cut,   lofar_fp   = reproject_interp((lofar_data,   wcs_L), wcs_out, shape_out=(nx, ny))

#### analysis
plt.imshow(meerkat_cut, vmin=-np.nanstd(meerkat_cut), vmax=5*np.nanstd(meerkat_cut), origin='lower')
plt.colorbar()
plt.show()

plt.imshow(lofar_cut, vmin=-np.nanstd(lofar_cut), vmax=5*np.nanstd(lofar_cut), origin='lower')
plt.colorbar()
plt.show()

# spectral index map
spx = spectral_index(meerkat_cut, lofar_cut, 1359.7, 150)
plt.imshow(spx, origin='lower', vmin=-2, vmax=2, cmap='bwr')
plt.colorbar()
plt.show()
