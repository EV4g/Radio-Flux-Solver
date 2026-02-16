from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from reproject import reproject_interp
import matplotlib.pyplot as plt
import glob

basedir = "/home/floris/Documents/PhD/Galactic plane/"

meerkat_files = np.sort(glob.glob(basedir+"LOFAR_and_MeerKAT/data/smgps_mfs_images/*.fits"))
#lofar_files = np.sort(glob.glob(basedir+"P282+00/*.fits"))
#lofar_files = np.sort(glob.glob(basedir+"P282+00/ddf/Multi/HogbomMulti/*.fits"))

lofar_files = np.sort(glob.glob("/home/floris/Downloads/low-mosaic-blanked.fits")) # low-res

meerkat_file = meerkat_files[11]
lofar_file = lofar_files[0] #3

# load meerkat data
mhdul = fits.open(meerkat_file)
meerkat_data = mhdul[0].data[0, 0]
meerkat_header = mhdul[0].header
wcs_M = WCS(meerkat_header).celestial

# load lofar file
lhdul = fits.open(lofar_file)
lofar_data = lhdul[0].data if len(lhdul[0].data.shape) == 2 else lhdul[0].data[0, 0]
lofar_header = lhdul[0].header
wcs_L = WCS(lofar_header).celestial

# cutout (galactic lat,lon)
cutout_lat = [-0.8, 0]
cutout_lon = [34.2, 35]

l0 = np.mean(cutout_lon) * u.deg
b0 = np.mean(cutout_lat) * u.deg
center = SkyCoord(l=l0, b=b0, frame="galactic")

dlon = (cutout_lon[1] - cutout_lon[0]) * u.deg
dlat = (cutout_lat[1] - cutout_lat[0]) * u.deg
width  = abs(dlon)
height = abs(dlat)

# choose an output pixel scale (pick the finer of the two so you don't lose detail)
# cdelt is deg/pix; take absolute and min across both images
scale_M = np.min(np.abs(wcs_M.wcs.cdelt)) * u.deg
scale_L = np.min(np.abs(wcs_L.wcs.cdelt)) * u.deg
pixscale = max(scale_M, scale_L)

nx = int(np.ceil((width  / pixscale).decompose().value))
ny = int(np.ceil((height / pixscale).decompose().value))

# build WCS
wcs_out = WCS(naxis=2)
wcs_out.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
wcs_out.wcs.cunit = ["deg", "deg"]
wcs_out.wcs.crval = [center.galactic.l.deg, center.galactic.b.deg]
wcs_out.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
wcs_out.wcs.cdelt = [-pixscale.to_value(u.deg), pixscale.to_value(u.deg)]

shape_out = (ny, nx)

# reproject both to the same grid
meerkat_cut, meerkat_fp = reproject_interp((meerkat_data, wcs_M), wcs_out, shape_out=shape_out)
lofar_cut,   lofar_fp   = reproject_interp((lofar_data,   wcs_L), wcs_out, shape_out=shape_out)




#### analysis

def spectral_index(S1, S2, v1, v2):
    return (np.log(S1) - np.log(S2)) / (np.log(v1) - np.log(v2))

plt.imshow(meerkat_cut, vmin=-0.005, vmax=0.02, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(lofar_cut, vmin=-0.005, vmax=0.02, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(spectral_index(meerkat_cut, lofar_cut, 1359.7, 150), origin='lower', vmin=-2, vmax=2, cmap='bwr')
plt.colorbar()
