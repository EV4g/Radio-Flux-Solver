from concurrent.futures import ProcessPoolExecutor
from functools import partial
from astropy.nddata.utils import Cutout2D
from reproject import reproject_interp
from tqdm import tqdm
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np

""""""
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

""""""
def generate_new_wcs(center, nx, ny, pixscale):
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    wcs_out.wcs.cunit = ["deg", "deg"]
    wcs_out.wcs.crval = [center.galactic.l.deg, center.galactic.b.deg]
    wcs_out.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs_out.wcs.cdelt = [-pixscale.to_value(u.deg), pixscale.to_value(u.deg)]
    return wcs_out

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

    try:    beam_area_1 = (np.pi / (4*np.log(2))) * (header1['BMAJ'] / pix_per_deg.value) * (header1['BMIN'] / pix_per_deg.value)
    except: beam_area_1 = (np.pi / (4*np.log(2))) * (header1['CLEANBMJ'] / pix_per_deg.value) * (header1['CLEANBMN'] / pix_per_deg.value)

    try:    beam_area_2 = (np.pi / (4*np.log(2))) * (header2['BMAJ'] / pix_per_deg.value) * (header2['BMIN'] / pix_per_deg.value)
    except: beam_area_2 = (np.pi / (4*np.log(2))) * (header2['CLEANBMJ'] / pix_per_deg.value) * (header2['CLEANBMN'] / pix_per_deg.value)

    pos = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")

    try:
        cutout1 = Cutout2D(data1, position=pos, size=(h, w), wcs=wcs1, mode="partial", fill_value=np.nan)
        cutout2 = Cutout2D(data2, position=pos, size=(h, w), wcs=wcs2, mode="partial", fill_value=np.nan)
    except: return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


    if np.isnan(cutout1.data).all() or np.isnan(cutout2.data).all():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        wcs_out = generate_new_wcs(pos, nx, ny, pix_per_deg)

        try:
            reproj1, _ = reproject_interp((cutout1.data, cutout1.wcs), wcs_out, shape_out=(nx, ny))
            reproj2, _ = reproject_interp((cutout2.data, cutout2.wcs), wcs_out, shape_out=(nx, ny))
        except: return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

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