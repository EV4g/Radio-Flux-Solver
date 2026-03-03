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

from functions import get_survey_file

warnings.filterwarnings("ignore", category=FITSFixedWarning)
basedir = "/home/floris/Documents/PhD/Galactic plane/LOFAR_and_MeerKAT/data/"

meerkat_files = np.sort(glob.glob(basedir+"meerkat/*.fits"))
racs_files = np.sort(glob.glob(basedir+"racs/*.fits"))
vlssr_files = np.sort(glob.glob(basedir+"vlssr/*.fits"))
lofar_files = np.sort(glob.glob(basedir+"lofar/*.fits"))


pos = SkyCoord(282.00*u.deg, 0.00*u.deg, frame="icrs")


print(get_survey_file(meerkat_files, pos))
print(get_survey_file(racs_files, pos))
print(get_survey_file(vlssr_files, pos))
