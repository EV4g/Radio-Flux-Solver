from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import multiprocessing; multiprocessing.set_start_method('fork') #for windows/mac
from functions import get_flux_batch, prep_file, compute_fluxcal_statistics
from astropy.table import Table

"""Plot (log)ratio as function of position in field."""
def plot_location_dependant_index(ra, dec, ratio):
    plt.scatter(ra, dec, c=ratio)
    plt.colorbar()
    plt.gca().set_box_aspect(1)
    plt.show()

def sources_in_fits(ra_deg, dec_deg, fn):
    """
    Given arrays of RA/Dec (degrees) and a FITS file,
    return a boolean array of which sources fall within the image footprint.
    """
    with fits.open(fn) as hdul:
        w  = WCS(hdul[0].header).celestial
        nx = hdul[0].header["NAXIS1"]
        ny = hdul[0].header["NAXIS2"]

    coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit=u.deg, frame="icrs")
    x, y   = w.world_to_pixel(coords)

    return (x >= 0) & (x < nx) & (y >= 0) & (y < ny)


spectral_index_theory = -0.7
lofar_freq = 144.6e6 #Hz
racs_freq = 887.5e6 #Hz
meerkat_freq = 1367e6 #Hz

lofar_files = glob.glob(os.getcwd()+"/data/lofar/*.fits")[0]

# get calagogs
racs    = Table.read(os.getcwd()+"/catalogs/racs/racs_clean.csv")
meerkat = Table.read(os.getcwd()+"/catalogs/meerkat/meerkat_clean.csv")
vlssr   = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_clean.csv")

# check for sources in current file
racs_valid      = sources_in_fits(racs['ra'],    racs['dec'],    lofar_files)
meerkat_valid   = sources_in_fits(meerkat['ra'], meerkat['dec'], lofar_files)
vlssr_valid     = sources_in_fits(vlssr['ra'],   vlssr['dec'],   lofar_files)

# catalog plot
plt.hist(np.log10(racs['flux_jy']), alpha=0.6, bins=100, label='racs')
plt.hist(np.log10(meerkat['flux_jy']), alpha=0.6, bins=100, label='meerkat')
plt.hist(np.log10(vlssr['flux_jy']), alpha=0.6, bins=100, label='vlssr')
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.show()

plt.hist(np.log10(racs['flux_jy'][racs_valid]), alpha=0.6, bins=20, label='racs_v')
plt.hist(np.log10(meerkat['flux_jy'][meerkat_valid]), alpha=0.6, bins=20, label='meerkat_v')
plt.hist(np.log10(vlssr['flux_jy'][vlssr_valid]), alpha=0.6, bins=20, label='vlssr_v')
plt.legend()
plt.xlabel("log10(flux/Jy)")
plt.ylabel("count")
plt.yscale('log')
plt.show()


# plot catalog as function of position
plt.scatter(racs['ra'][racs_valid], racs['dec'][racs_valid], s=2)
plt.scatter(meerkat['ra'][meerkat_valid], meerkat['dec'][meerkat_valid], s=2)
plt.scatter(vlssr['ra'][vlssr_valid], vlssr['dec'][vlssr_valid], s=2)
plt.gca().set_box_aspect(1)
plt.xlabel("RA")
plt.ylabel("Dec")
plt.show()

