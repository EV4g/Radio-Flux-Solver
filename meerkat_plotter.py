from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import glob

basedir = "/home/floris/Documents/PhD/Galactic plane/LOFAR_and_MeerKAT/data/smgps_mfs_images/"

files = np.sort(glob.glob(basedir+"*.fits"))

file = files[23]

hdul = fits.open(file)
data = hdul[0].data
header = hdul[0].header