import numpy as np
import glob
from functions import get_overlapping_files

basedir = "/home/floris/Documents/PhD/Galactic plane/LOFAR_and_MeerKAT/data/"

meerkat_files = np.sort(glob.glob(basedir+"meerkat/*.fits"))
racs_files = np.sort(glob.glob(basedir+"racs/*.fits"))
vlssr_files = np.sort(glob.glob(basedir+"vlssr/*.fits"))
lofar_files = np.sort(glob.glob(basedir+"lofar/*.fits"))

print(get_overlapping_files(lofar_files[0], meerkat_files), "\n")
print(get_overlapping_files(lofar_files[0], racs_files), "\n")
print(get_overlapping_files(lofar_files[0], vlssr_files), "\n")