from concurrent.futures import ProcessPoolExecutor
from functools import partial
from astropy.nddata.utils import Cutout2D
from reproject import reproject_interp
from tqdm import tqdm
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt

"""Get two catalogs and return flux and snr (flux / e_flux)"""
def get_catalog_matched_flux(cat1, cat2, thres_arc=2):
    idx_cat1, idx_cat2 = match_catalogs_2D([cat1, cat2], thres_arc=thres_arc)

    flux1 = cat1.flux[idx_cat1]
    snr1 = flux1 / cat1.e_flux[idx_cat1]

    flux2 = cat2.flux[idx_cat2]
    snr2 = flux2 / cat2.e_flux[idx_cat2]

    return flux1, flux2, snr1, snr2

"""Load two catalogs and plot their relative fluxes, according to a powerlaw"""
def quick_compare_catalog(cat1, cat2, config):
    flux1, flux2, _, _ = get_catalog_matched_flux(cat1, cat2, thres_arc=config.thres_arc)
    spectral_flux_ratio, spectral_index_actual, x, log_ratio, scale_factor = compute_fluxcal_statistics(cat1.freq, cat2.freq, flux1, flux2, config.spectral_index_theory)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(flux2, flux1, s=10, alpha=0.7)
    ax[0].plot(x, x * spectral_index_actual, color='purple', ls='--', label="Data fit")
    ax[0].plot(x, x, color='black', ls='--', label="x = y")
    ax[0].plot(x, x * spectral_flux_ratio, 'r--', label=f'Expected (α={config.spectral_index_theory})')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(f"{cat1.name} flux [Jy]"); ax[0].set_ylabel(f"{cat2.name} flux [Jy]")
    ax[0].legend()
    
    # Log-ratio histogram
    ax[1].hist(log_ratio, bins=30, edgecolor='k')
    ax[1].axvline(scale_factor, color='red', ls='--', label=f'Median = {scale_factor:.3f}')
    ax[1].set_xlabel(f"log10(S_{cat1.name} / S_{cat2.name})")
    ax[1].set_ylabel("N")
    ax[1].legend()
    plt.tight_layout()
    #plt.savefig("flux_scale_comparison.png", dpi=150)
    plt.show()

    print(f"Compared {cat1.name} to {cat2.name} \n")

"""Linspace, but logarithmic scaling to avoid oversampling at low values"""
def log_linspace(mn, mx, n):
    return 10**np.linspace(np.log10(mn), np.log10(mx), n)

"""Remove outliers at eitherside of the distribution"""
def remove_outliers(var, clip_percentage):
    clip_top = np.percentile(var, 100 - clip_percentage)
    clip_bottom = np.percentile(var, clip_percentage)
    return var[(var < clip_top) & (var > clip_bottom)]

"""Remove outliers at eitherside of the distribution, consider two variables at once"""
def remove_outliers_2D(var1, var2, clip_percentage):
    v1_clip_top = np.percentile(var1, 100 - clip_percentage)
    v1_clip_bottom = np.percentile(var1, clip_percentage)
    v2_clip_top = np.percentile(var2, 100 - clip_percentage)
    v2_clip_bottom = np.percentile(var2, clip_percentage)

    condition = (var1 > v1_clip_bottom) & (var1 < v1_clip_top) & (var2 > v2_clip_bottom) & (var2 < v2_clip_top)

    return var1[condition], var2[condition]

"""Remove outliers at eitherside of the distribution, consider two variables at once, repeat n times"""
def remove_outliers_2D_iterative(var1, var2, clip_percentage, n):
    for _ in range(n):
        v1_clip_top = np.percentile(var1, 100 - clip_percentage)
        v1_clip_bottom = np.percentile(var1, clip_percentage)
        v2_clip_top = np.percentile(var2, 100 - clip_percentage)
        v2_clip_bottom = np.percentile(var2, clip_percentage)

        condition = (var1 > v1_clip_bottom) & (var1 < v1_clip_top) & (var2 > v2_clip_bottom) & (var2 < v2_clip_top)
        var1 = var1[condition]
        var2 = var2[condition]

    return var1, var2

"""Get frequency and flux arrays for two different datasets, and compute the flux-calibration offset between them compared to the -0.7 spectral index baseline
data_1 gets compared versus the baseline of data_2. Returns (expected) fitted spectral_flux_ratio, and (actual) index, as well as the (offset) ratios and scaling factor.
Optional Valid parameter to only select a subset of all arrays."""
def compute_fluxcal_statistics(freq1, freq2, flux1, flux2, spectral_index_theory=-0.7, valid=None):
    if type(valid) is not type(None):
        flux1 = flux1[valid]
        flux2 = flux2[valid]
    spectral_flux_ratio = (freq1 / freq2)**spectral_index_theory
    spectral_index_actual = 10**np.mean(np.log10(flux1 / flux2))
    x = log_linspace(np.min(flux2), np.max(flux2), 10)

    ratio = flux1 / (flux2 * spectral_flux_ratio)
    #valid_ratio = np.isfinite(ratio) & (ratio > 0)
    log_ratio = np.log10(ratio)

    scale_factor = np.median(log_ratio)
    scatter      = np.std(log_ratio)
    N            = len(log_ratio)
    stderr       = scatter / np.sqrt(N)

    print(f"N valid sources      : {N}")
    print(f"Median log10(ratio)  : {scale_factor:.4f}  ({10**scale_factor:.4f}×)")
    print(f"Scatter (1σ)         : {scatter:.4f} dex")
    print(f"Uncertainty on median: ±{stderr:.4f} dex")

    return spectral_flux_ratio, spectral_index_actual, x, log_ratio, scale_factor

"""Plot (log)ratio as function of position in field."""
def plot_location_dependant_index(ra, dec, ratio):
    plt.scatter(ra, dec, c=ratio)
    plt.colorbar()
    plt.gca().set_box_aspect(1)
    plt.show()

"""(cat1, cat2) --> [(ra1, dec1), (ra2, dec2)]"""
def radec_list(cats):
    radec_list = []
    for cat in cats:
        radec_list.append((cat.ra, cat.dec))
    return radec_list

"""(cat1, cat2) --> [(ra1, dec1), (ra2, dec2)]"""
def radec_list_simple(cats):
    radec_list = []
    for cat in cats:
        radec_list.append((cat['ra'], cat['dec']))
    return radec_list

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
    if sy is None: return A * sx**2 * 2 * np.pi
    else:        return A * sx * sy * 2 * np.pi

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
