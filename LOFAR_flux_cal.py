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
import os
from astropy.nddata.utils import Cutout2D

from functions import spectral_index, get_pixscale, generate_new_wcs

racs_files = glob.glob(os.getcwd()+"/data/racs/*.fits")
lofar_files = np.sort(glob.glob("/home/floris/Documents/PhD/Galactic plane/P282+00/ddf/Multi/MSC_deep/*.fits")) #1 restored; 3 dirty
cat = fits.open(os.getcwd()+"/data/P282+00.offset_cat.fits")[1]

# load racs file
racs_file = racs_files[2]
hdul = fits.open(racs_file)
racs_data = hdul[0].data[0, 0]
wcs_R = WCS(hdul[0].header).celestial

# load lofar file
lofar_file = lofar_files[1]
lhdul = fits.open(lofar_file)
lofar_data = lhdul[0].data if len(lhdul[0].data.shape) == 2 else lhdul[0].data[0, 0]
wcs_L = WCS(lhdul[0].header).celestial

# get calagogue ra, dec
cat_ra, cat_dec = cat.data['RA'], cat.data['DEC']

from time import perf_counter

start = perf_counter()

w, h = 1.5 * u.arcmin, 1.5 * u.arcmin
pixscale, nx, ny = get_pixscale(wcs_R, wcs_L, w, h)
for i, (ra, dec) in enumerate(zip(cat_ra, cat_dec)):    
    pos = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")

    # cutout each image around (ra,dec) in its native grid (fast)
    r_co = Cutout2D(racs_data,  position=pos, size=(h, w), wcs=wcs_R, mode="partial", fill_value=np.nan)
    l_co = Cutout2D(lofar_data, position=pos, size=(h, w), wcs=wcs_L, mode="partial", fill_value=np.nan)
    
    # build common output WCS centered on (ra,dec) with your chosen nx,ny,pixscale
    wcs_out = generate_new_wcs(pos, nx, ny, pixscale)
    
    # reproject both cutouts onto the same grid
    racs_cut, racs_fp = reproject_interp((r_co.data, r_co.wcs), wcs_out, shape_out=(nx, ny))
    lofar_cut, lofar_fp = reproject_interp((l_co.data, l_co.wcs), wcs_out, shape_out=(nx, ny))
    
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].imshow(racs_cut, origin="lower", vmin=-np.nanstd(racs_cut), vmax=5*np.nanstd(racs_cut)); ax[0].set_title("RACS")
    ax[1].imshow(lofar_cut, origin="lower", vmin=-np.nanstd(lofar_cut), vmax=5*np.nanstd(lofar_cut)); ax[1].set_title("LOFAR")
    plt.suptitle(str(i))
    plt.show()





#### analysis
# plt.imshow(racs_cut, vmin=-np.nanstd(racs_cut), vmax=5*np.nanstd(racs_cut), origin='lower')
# plt.colorbar()
# plt.show()

# plt.imshow(lofar_cut, vmin=-np.nanstd(lofar_cut), vmax=5*np.nanstd(lofar_cut), origin='lower')
# plt.colorbar()
# plt.show()


