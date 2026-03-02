import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from astropy.nddata.utils import Cutout2D
from reproject import reproject_interp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FITSFixedWarning)

"""Load fits file and extract data, header, wcs"""
def prep_file(file):
    hdul = fits.open(file)
    data = hdul[0].data if len(hdul[0].data.shape) == 2 else hdul[0].data[0, 0]
    header = hdul[0].header
    wcs = WCS(header).celestial
    return data, header, wcs

"""List all meerkat files in dir that overlap with the given coordinate"""
def get_meerkat_file(files, coord):
    for i, fn in enumerate(files):

        with fits.open(fn) as hdul:
            hdr = hdul[0].header
            w = WCS(hdr).celestial
            nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]

        # Convert input coord into wcs frame and then into pix
        c = coord.transform_to(w.wcs.radesys.lower() if w.wcs.radesys else "icrs")
        x, y = w.world_to_pixel(c)

        if (0 <= x < nx) and (0 <= y < ny): return files[i]

    return None

"""Get spectral index based on two fluxes and two frequencies"""
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


def get_pixscale(wcs_A, wcs_B, w, h, use_highest_res=True):
    # choose an output pixel scale (pick the finer of the two so you don't lose detail)
    # cdelt is deg/pix; take absolute and min across both images
    scale_M = np.min(np.abs(wcs_A.wcs.cdelt)) * u.deg
    scale_L = np.min(np.abs(wcs_B.wcs.cdelt)) * u.deg
    
    if use_highest_res: pixscale = min(scale_M, scale_L)
    else: pixscale = max(scale_M, scale_L)

    nx = int(np.ceil((w  / pixscale).decompose().value))
    ny = int(np.ceil((h / pixscale).decompose().value))
    return pixscale, nx, ny
    

def generate_new_wcs(center, nx, ny, pixscale):
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    wcs_out.wcs.cunit = ["deg", "deg"]
    wcs_out.wcs.crval = [center.galactic.l.deg, center.galactic.b.deg]
    wcs_out.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs_out.wcs.cdelt = [-pixscale.to_value(u.deg), pixscale.to_value(u.deg)]
    return wcs_out

"""A 2D gaussian function, used for fitting point sources"""
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):    
    x, y = xy
    
    dx, dy = x - xo, y - yo
    sint_sq, cost_sq = np.sin(theta)**2, np.cos(theta)**2
    two_sig_x_sq, two_sig_y_sq = 2*sigma_x**2, 2*sigma_y**2
    
    a = cost_sq / two_sig_x_sq + sint_sq / two_sig_y_sq
    b = -np.sin(2*theta) / (2 * two_sig_x_sq) + np.sin(2*theta) / (2 * two_sig_y_sq)
    c = sint_sq / two_sig_x_sq + cost_sq / two_sig_y_sq
    g = offset + amplitude*np.exp( - (a*dx**2 + 2*b*dx*dy + c*dy**2))
    return g.T.ravel()

"""A simplified 2D gaussian function, used for fitting point sources"""
def twoD_Gaussian_simple(xy, amplitude, xo, yo, sigma, offset):    
    x, y = xy
    dx, dy = x - xo, y - yo
    a = 1 / (2 * sigma**2)
    g = offset + amplitude*np.exp( - a*(dx**2 + dy**2))
    return g.T.ravel()

"""Function to automatically fit a gaussian, and return the best fit"""
def fit_gauss(array, simple=False, mf=1000, debug=False):
    xy = np.where(np.isfinite(array) & (array != 0))
    
    x = np.linspace(0, array.shape[0] - 1, array.shape[0])
    y = np.linspace(0, array.shape[1] - 1, array.shape[1])
    x, y = np.meshgrid(x, y)
    
    mx, mn = np.nanmax(array),  np.nanmin(array)
    diff = mx - mn 
    half_x, half_y = array.shape[0]/2, array.shape[1]/2
    
    if debug or simple:
        # amplitude, x, y, sigma, z_offset
        init = (mx, half_x, half_y, 1, mn)
        bounds = [[mn, 0, 0, 0, -np.inf], [mx, array.shape[0], array.shape[1], np.inf, mx]]
        popt, pcov = curve_fit(twoD_Gaussian_simple, xy, array[xy], p0=init, bounds=bounds, maxfev=mf)
        
        if debug: return twoD_Gaussian_simple((x, y), *popt).reshape((array.shape[0], array.shape[1])), popt, pcov
        else:     return twoD_Gaussian_simple((x, y), *popt).reshape((array.shape[0], array.shape[1]))
        
    else:
        # amplitude, x, y, sigma_x, sigma_y, theta, z_offset
        init = (diff, half_x, half_y, 1, 1, 0, 0)
        bounds = [[0, half_x-10, half_y-10, 0, 0, 0, -np.inf], [2 * diff, half_x+10, half_y+10, np.inf, np.inf, 2*np.pi, mx]]
        popt, _ = curve_fit(twoD_Gaussian, xy, array[xy], p0=init, bounds=bounds, maxfev=mf)
        return twoD_Gaussian((x, y), *popt).reshape((array.shape[0], array.shape[1]))
    
"""Volume under a 2D gaussian """
def gaussian_volume(A, sx, sy=None):
    if sy==None: return A * sx**2 * 2 * np.pi
    else:        return A * sx * sy * 2 * np.pi
    
"""Helper function required for get_flux()"""
_get_flux_fixed = None
def _worker(args):
    ra, dec = args
    return _get_flux_fixed(ra, dec)

"""Gets the flux and other statistics from a (ra,dec) coordinate for two dataset files"""
def get_flux(w, h, data1, data2, header1, header2, ra, dec):
    wcs1 = WCS(header1).celestial
    wcs2 = WCS(header2).celestial
    pix_per_deg, nx, ny = get_pixscale(wcs1, wcs2, w, h)

    beam_area_1 = (np.pi / (4*np.log(2))) * (header1['BMAJ'] / pix_per_deg.value) * (header1['BMIN'] / pix_per_deg.value)
    beam_area_2 = (np.pi / (4*np.log(2))) * (header2['BMAJ'] / pix_per_deg.value) * (header2['BMIN'] / pix_per_deg.value)

    pos = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")

    cutout1 = Cutout2D(data1, position=pos, size=(h, w), wcs=wcs1, mode="partial", fill_value=np.nan)
    cutout2 = Cutout2D(data2, position=pos, size=(h, w), wcs=wcs2, mode="partial", fill_value=np.nan)

    wcs_out = generate_new_wcs(pos, nx, ny, pix_per_deg)

    reproj1, _ = reproject_interp((cutout1.data, cutout1.wcs), wcs_out, shape_out=(nx, ny))
    reproj2, _ = reproject_interp((cutout2.data, cutout2.wcs), wcs_out, shape_out=(nx, ny))

    fit1, popt1, pcov1 = fit_gauss(reproj1, simple=True, debug=True)
    fit2, popt2, pcov2 = fit_gauss(reproj2, simple=True, debug=True)

    local_snr     = np.nanmax(reproj1) / np.nanstd(reproj1)
    local_snr_fit = np.nanmax(reproj2) / np.nanstd(reproj2)

    sigma_pcov1 = np.sqrt(pcov1[3, 3])
    sigma_pcov2 = np.sqrt(pcov2[3, 3])

    dist = np.sqrt((popt1[1] - popt2[1])**2 + (popt1[2] - popt2[2])**2)

    flux1 = gaussian_volume(popt1[0], popt1[3]) / beam_area_1
    flux2 = gaussian_volume(popt2[0], popt2[3]) / beam_area_2

    return flux1, flux2, dist, sigma_pcov1, sigma_pcov2, local_snr, local_snr_fit

"""Multithread the get_flux() function"""
def get_flux_batch(w, h, data1, data2, header1, header2, ra, dec, max_workers=24, chunksize=8):
    global _get_flux_fixed
    _get_flux_fixed = partial(get_flux, w, h, data1, data2, header1, header2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_worker, zip(ra, dec), chunksize=chunksize), total=len(ra)))

    return map(np.array, zip(*results))