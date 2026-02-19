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

warnings.filterwarnings("ignore", category=FITSFixedWarning)


def get_meerkat_file(files, coord):
    for i, fn in enumerate(files):

        with fits.open(fn) as hdul:
            hdr = hdul[0].header
            w = WCS(hdr).celestial
            nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]

        # Convert input coord into wcs frame and then into pix
        c = coord.transform_to(w.wcs.radesys.lower() if w.wcs.radesys else "icrs")
        x, y = w.world_to_pixel(c)

        if (0 <= x < nx) and (0 <= y < ny): return i

    return None

def spectral_index(S1, S2, v1, v2):
    return (np.log(S1) - np.log(S2)) / (np.log(v1) - np.log(v2))

def cutout_to_galactic_wh(cutout_lon, cutout_lat):
    l0 = np.mean(cutout_lon) * u.deg
    b0 = np.mean(cutout_lat) * u.deg
    center = SkyCoord(l=l0, b=b0, frame="galactic")
    
    dlon = (cutout_lon[1] - cutout_lon[0]) * u.deg
    dlat = (cutout_lat[1] - cutout_lat[0]) * u.deg
    width  = abs(dlon)
    height = abs(dlat)
    return center, width, height


def get_pixscale(wcs_A, wcs_B, w, h):
    # choose an output pixel scale (pick the finer of the two so you don't lose detail)
    # cdelt is deg/pix; take absolute and min across both images
    scale_M = np.min(np.abs(wcs_A.wcs.cdelt)) * u.deg
    scale_L = np.min(np.abs(wcs_B.wcs.cdelt)) * u.deg
    pixscale = max(scale_M, scale_L)

    nx = int(np.ceil((w  / pixscale).decompose().value))
    ny = int(np.ceil((h / pixscale).decompose().value))
    return nx, ny
    

def generate_new_wcs(center, nx, ny, pixscale):
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    wcs_out.wcs.cunit = ["deg", "deg"]
    wcs_out.wcs.crval = [center.galactic.l.deg, center.galactic.b.deg]
    wcs_out.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs_out.wcs.cdelt = [-pixscale.to_value(u.deg), pixscale.to_value(u.deg)]
    return wcs_out