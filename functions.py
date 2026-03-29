import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit, minimize
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from astropy.nddata.utils import Cutout2D
from reproject import reproject_interp
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from itertools import combinations
from scipy.stats import chi2 as _chi2_dist, binned_statistic_2d

warnings.filterwarnings("ignore", category=FITSFixedWarning)

"""Linspace, but logarithmic scaling to avoid oversampling at low values"""
def log_linspace(mn, mx, n):
    return 10**np.linspace(np.log10(mn), np.log10(mx), n)

"""Average of an array in logspace"""
def log_average(arr, w):
    return np.exp(np.average(np.log(arr), weights=w))

"""Load fits file and extract data, header, wcs"""
def prep_file(file):
    hdul = fits.open(file)
    data = hdul[0].data if len(hdul[0].data.shape) == 2 else hdul[0].data[0, 0]
    header = hdul[0].header
    wcs = WCS(header).celestial
    return data, header, wcs

"""List all meerkat files in dir that overlap with the given coordinate"""
def get_survey_file(files, coord):
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

"""Return all corners from one .fits file"""
def get_corners(fn):
    with fits.open(fn) as hdul:
        hdr = hdul[0].header
        w = WCS(hdr).celestial
        nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
    corners_pix = np.array([[0, 0], [nx-1, 0], [0, ny-1], [nx-1, ny-1]], dtype=float)
    corners_sky = w.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
    return w, nx, ny, corners_sky

"""Return whether or not the corners from one .fits file (corners_sky) are within the bounds of another .fits file (w, nx, ny)"""
def any_corner_inside(corners_sky, w, nx, ny):
    xs, ys = w.world_to_pixel(corners_sky)
    return bool(np.any((xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)))

"""Check one .fits file against a host of others and return the files that overlap"""
def get_overlapping_files(ref_file, files):
    ref_w, ref_nx, ref_ny, ref_corners = get_corners(ref_file)

    overlapping = []
    for fn in files:
        if fn == ref_file:
            continue
        try:
            w, nx, ny, corners = get_corners(fn)
            if any_corner_inside(ref_corners, w, nx, ny) or any_corner_inside(corners, ref_w, ref_nx, ref_ny):
                overlapping.append(fn)
        except Exception as e:
            print(f"Skipping {fn}: {e}")

    return overlapping

"""Get spectral index based on two fluxes and two frequencies"""
def get_spectral_index(S1, S2, v1, v2):
    return (np.log(S1) - np.log(S2)) / (np.log(v1) - np.log(v2))

""""""
def cutout_to_galactic_wh(cutout_lon, cutout_lat):
    l0 = np.mean(cutout_lon) * u.deg
    b0 = np.mean(cutout_lat) * u.deg
    center = SkyCoord(l=l0, b=b0, frame="galactic")

    dlon = (cutout_lon[1] - cutout_lon[0]) * u.deg
    dlat = (cutout_lat[1] - cutout_lat[0]) * u.deg
    width  = abs(dlon)
    height = abs(dlat)
    return center, width, height

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

"""Return beam size [degree] from fits header"""
def get_beam_size(file):
    hdul = fits.open(file)
    header = hdul[0].header
    try:
        return header['BMAJ'], header['BMIN'], header['BPA']
    except:
        return header['CLEANBMJ'], header['CLEANBMN'], header['CLEANBPA']
    return None

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

"""Return a per-source 1D positional RMS (deg) for a catalog.
e_ra / e_dec are combined as sigma_1d = sqrt((e_ra^2 + e_dec^2) / 2)."""
def get_pos_err_deg(cat):
    return np.sqrt((cat.e_ra ** 2 + cat.e_dec ** 2) / 2.0) # degrees

def radec_to_xyz(ra_deg, dec_deg):
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cd  = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])

def match_catalogs_2D(cat_list, thres_arc=2, nsigma=3.0, crowd_radius_arc=None, anchor_index=0, return_quality=False, thres_arc_override=False):
    """Fast n-catalogue cross-matcher with adaptive positional uncertainties,
    crowding detection, and per-pair quality metrics.
    
    Parameters
    ----------
    cat_list           : list of catalogue objects for ra,dec [degrees] and flux, and .e_flux.
    thres_arc          : fallback fixed match radius [arcsec]
    nsigma             : n * sigmas used as the acceptance radius for point matching
    crowd_radius_arc   : if given, count same-catalogue neighbours within this
                         radius [arcsec] per source and store in quality['n_crowd']
    anchor_index       : catalogue index that anchors the multi-catalogue search
    return_quality     : if True, also return quality metrics
    thres_arc_override : if True, forces matching to use thres_arc matching instead of 
                         error-based nsigma * err_rad
    
    Returns
    ----------------
        [idx_0, ..., idx_{n-1}]  [, quality]
    
    quality metrics
    ------------
        'sep_arcsec'  {(a,b): ndarray}  arcsec separation for each matched pair,
                      in the same order as the returned index arrays.
        'p_match'     {(a,b): ndarray}  per-pair match probability from a
                      chi-squared positional test.
        'n_crowd'     {cat_index: ndarray}  number of same-catalogue neighbours
                      within crowd_radius_arc for each source.  Only populated
                      when crowd_radius_arc is given.
    """

    n = len(cat_list)

    # Precompute vectors and errors per catalog
    xyz_all  = []
    errs     = []
    for cat in cat_list:
        xyz_all.append(radec_to_xyz(cat.ra, cat.dec))
        if not thres_arc_override:
            errs.append(cat.err_rad)
        else:
            errs.append(None)
    use_errs = not thres_arc_override

    # Prebuild one KD-tree per catalog
    trees = [cKDTree(xyz) if len(xyz) > 0 else None for xyz in xyz_all]

    # Crowding in 3D (no projection needed)
    crowd_counts = {}
    if crowd_radius_arc is not None:
        crowd_r_3d = 2.0 * np.sin(np.deg2rad(crowd_radius_arc / 3600.0) / 2.0)
        for i, (xyz, tree) in enumerate(zip(xyz_all, trees)):
            if tree is None:
                crowd_counts[i] = np.array([], dtype=int)
                continue
            nbrs = tree.query_ball_point(xyz, r=crowd_r_3d, workers=-1)
            crowd_counts[i] = np.array([len(nb) - 1 for nb in nbrs])

    matched_results = {}
    sep_results     = {}
    prob_results    = {}
    pair_maps       = {}

    for a in range(n):
        for b in range(a + 1, n):
            xyz_a = xyz_all[a]
            xyz_b = xyz_all[b]

            if len(xyz_a) == 0 or len(xyz_b) == 0:
                matched_results[(a, b)] = ([], [])
                sep_results[(a, b)]     = np.array([])
                prob_results[(a, b)]    = np.array([])
                pair_maps[(a, b)]       = {}
                pair_maps[(b, a)]       = {}
                continue

            # Always query smaller set against larger tree
            if len(xyz_a) >= len(xyz_b):
                sub_xyz, sup_err, sub_err, tree_sup, normal = xyz_b, errs[a], errs[b], trees[a], True
            else:
                sub_xyz, sup_err, sub_err, tree_sup, normal = xyz_a, errs[b], errs[a], trees[b], False

            # Coarse search
            if use_errs:
                sigma_pair   = np.hypot(np.median(sup_err), np.median(sub_err))
                query_radius = 2.0 * np.sin(nsigma * sigma_pair / 2.0)
            else:
                query_radius = 2.0 * np.sin(np.deg2rad(thres_arc / 3600.0) / 2.0)

            dists, idxs = tree_sup.query(sub_xyz, k=1, distance_upper_bound=query_radius, workers=-1)

            # Per-source refinement
            if use_errs:
                prelim = dists < query_radius
                accept = np.zeros(len(sub_xyz), dtype=bool)
                pidx   = np.where(prelim)[0]
                if len(pidx):
                    combined_sig = np.hypot(sub_err[pidx], sup_err[idxs[pidx]])
                    thresh_3d    = 2.0 * np.sin(nsigma * combined_sig / 2.0)
                    accept[pidx] = dists[pidx] < thresh_3d
                valid = accept
            else:
                valid = dists < query_radius

            matched_sub   = np.where(valid)[0]
            matched_sup   = idxs[valid]
            matched_dists = dists[valid]

            # Deduplication: keep closest sub per sup
            if len(matched_sup) != len(np.unique(matched_sup)):
                order             = np.lexsort((matched_dists, matched_sup))
                matched_sup       = matched_sup[order]
                matched_sub       = matched_sub[order]
                matched_dists     = matched_dists[order]
                first             = np.empty(len(matched_sup), dtype=bool)
                first[0]          = True
                first[1:]         = matched_sup[1:] != matched_sup[:-1]
                matched_sup       = matched_sup[first]
                matched_sub       = matched_sub[first]
                matched_dists     = matched_dists[first]

            # Convert distance to arcsec
            sep_arcsec = np.rad2deg(2.0 * np.arcsin(matched_dists / 2.0)) * 3600.0

            # Match probability via chi_sq on angular separation
            if use_errs:
                sep_rad        = np.deg2rad(sep_arcsec / 3600.0)
                combined_sig_f = np.hypot(sub_err[matched_sub], sup_err[matched_sup])
                chi2_vals      = (sep_rad / combined_sig_f) ** 2
                p_match        = 1.0 - _chi2_dist.cdf(chi2_vals, df=2)
            else:
                p_match = np.ones(len(matched_sub))

            if normal:
                idx_a_list = matched_sup.tolist()
                idx_b_list = matched_sub.tolist()
            else:
                idx_a_list = matched_sub.tolist()
                idx_b_list = matched_sup.tolist()

            matched_results[(a, b)] = (idx_a_list, idx_b_list)
            sep_results[(a, b)]     = sep_arcsec
            prob_results[(a, b)]    = p_match
            pair_maps[(a, b)]       = dict(zip(idx_b_list, idx_a_list))
            pair_maps[(b, a)]       = dict(zip(idx_a_list, idx_b_list))

    # Quality assessors
    quality = {
        'sep_arcsec': sep_results,
        'p_match':    prob_results if use_errs else {},
        'n_crowd':    crowd_counts,
    }

    # If only two catalogues, already done
    if n == 2:
        i0 = np.array(matched_results[(0, 1)][0])
        i1 = np.array(matched_results[(0, 1)][1])
        if return_quality:
            return (i0, i1), quality
        return (i0, i1)

    # If more than 2 catalogs: coalescence anchored on anchor_index
    match_dict = {i: {} for i in range(n)}
    for (a, b), (idx_a, idx_b) in matched_results.items():
        for i_a, i_b in zip(idx_a, idx_b):
            match_dict[a][i_a] = match_dict[a].get(i_a, []) + [(b, i_b)]
            match_dict[b][i_b] = match_dict[b].get(i_b, []) + [(a, i_a)]

    used_indices       = {i: set() for i in range(n)}
    consistent_matches = {i: [] for i in range(n)}

    for idx in match_dict[anchor_index]:
        if idx in used_indices[anchor_index]:
            continue

        group       = {anchor_index: idx}
        to_check    = list(match_dict[anchor_index][idx])
        valid_group = True

        while to_check and valid_group:
            curr_cat, curr_idx = to_check.pop()
            if curr_cat in group:
                if group[curr_cat] != curr_idx:
                    valid_group = False
                    break
                continue
            group[curr_cat] = curr_idx
            for next_cat, next_idx in match_dict[curr_cat][curr_idx]:
                if next_cat not in group:
                    to_check.append((next_cat, next_idx))

        if not valid_group or len(group) != n: continue

        group_valid = True
        for aa in range(n):
            for bb in range(aa + 1, n):
                if pair_maps.get((aa, bb), {}).get(group[bb], None) != group[aa]:
                    group_valid = False
                    break
            if not group_valid: break

        if group_valid:
            for cat_i, src_i in group.items():
                consistent_matches[cat_i].append(src_i)
                used_indices[cat_i].add(src_i)

    result = [np.array(consistent_matches[i]) for i in range(n)]
    if return_quality:
        return result, quality
    return result

"""Add contours to scatterplot
Takes x, y coordinated and a per-source weighting c. Can make the contour fitting work in logspace by using
(logy, logx). n sets the 2D resolution of the mesh."""
def calculate_contour_statistics(x, y, c, logx=False, logy=False, n=1000):
    if logx: x = np.log10(x)
    if logy: y = np.log10(y)

    xi = np.linspace(x.min(), x.max(), n)
    yi = np.linspace(y.min(), y.max(), n)
    Xi, Yi = np.meshgrid(xi, yi)

    # Fast histogram mapping
    H, _, _ = np.histogram2d(x, y, bins=n, weights=c, range=[[x.min(), x.max()], [y.min(), y.max()]])
    n_data  = len(x)
    sigma_x = (n_data**(-0.2) * np.std(x)) / ((x.max() - x.min()) / n)
    sigma_y = (n_data**(-0.2) * np.std(y)) / ((y.max() - y.min()) / n)
    Zi = gaussian_filter(H.T, sigma=(sigma_y, sigma_x))

    # Minimizer starting point
    peak_idx         = np.argmax(Zi)
    peak_x0, peak_y0 = Xi.ravel()[peak_idx], Yi.ravel()[peak_idx]

    # KDE on subset
    kde    = gaussian_kde(np.vstack([x, y]), weights=c)
    result = minimize(lambda p: -kde(p)[0], x0=[peak_x0, peak_y0], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-10})
    peak_x0, peak_y0 = result.x

    if logx: Xi = 10**Xi; peak_x0 = 10**peak_x0
    if logy: Yi = 10**Yi; peak_y0 = 10**peak_y0

    return Xi, Yi, Zi, peak_x0, peak_y0

"""A simple way to save fits files to png"""
def fits_to_png(fits_path, output_path, hdu_index=0, vmin=None, vmax=None, cmap="viridis"):
    with fits.open(fits_path) as hdul:
        data = hdul[hdu_index].data[0,0]

    plt.imsave(output_path, data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

"""Plot (log)ratio as function of position in field."""
def plot_location_dependant_index(ra, dec, ratio):
    plt.scatter(ra, dec, c=ratio)
    plt.colorbar()
    plt.gca().set_box_aspect(1)
    plt.show()

"""(cat1, cat2) --> [(ra1, dec1), (ra2, dec2)]"""
def radec_list_simple(cats):
    radec_list = []
    for cat in cats:
        radec_list.append((cat['ra'], cat['dec']))
    return radec_list

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

"""Return indices of catalog of all unique, non-double, <size> combinations with the condition f[i] < f[i+1]"""
def get_combinations(cats, size=3, required_index=None, skip_index=None):
    freqs = [cat.freq for cat in cats]
    indexed = sorted(enumerate(zip(freqs, cats)), key=lambda x: x[1][0])
    
    skip     = {skip_index}     if isinstance(skip_index,     int) else set(skip_index)     if skip_index     is not None else set()
    required = {required_index} if isinstance(required_index, int) else set(required_index) if required_index is not None else set()
    
    result = []
    for combo in combinations(indexed, size):
        indices = tuple(i for i, (f, _) in combo)
        freqs_c = [f for _, (f, _) in combo]

        if freqs_c != sorted(freqs_c):                                          continue
        if required and not required.issubset(indices):                         continue
        if skip     and any(i in skip for i in indices):                        continue

        result.append(indices)
    return result

"""Return indices of catalog (str) of all unique, non-double, threeway combinations with the condition f1 < f2 < f3"""
def get_triplet_combinations(cats, required_index=None, skip_index=None):
    freqs = []
    for cat in cats: freqs += [cat.freq]
    indexed = sorted(enumerate(zip(freqs, cats)), key=lambda x: x[1][0])
    return [(i1, i2, i3) for (i1, (f1, _)), (i2, (f2, _)), (i3, (f3, _)) in combinations(indexed, 3) if f1 < f2 < f3
        and (required_index is None or required_index in (i1, i2, i3)) and (skip_index is None or skip_index not in (i1, i2, i3))]

"""(cat1, cat2) --> [(ra1, dec1), (ra2, dec2)]"""
def radec_list(cats):
    radec_list = []
    for cat in cats:
        radec_list.append((cat.ra, cat.dec))
    return radec_list

""""""
def weighted_bin_stats(x, y, w, n_bins=200):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mn  = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (x >= edges[i]) & (x < edges[i + 1])
        if in_bin.sum() <= 2: continue
        y_b, w_b = y[in_bin], w[in_bin]
        w_sum = w_b.sum()
        if w_sum == 0: continue
        wmean    = np.dot(w_b, y_b) / w_sum
        mn[i]    = wmean
        std[i]   = np.sqrt(np.dot(w_b, (y_b - wmean) ** 2) / w_sum)

    ok = ~np.isnan(mn)
    return centers[ok], mn[ok], std[ok]

""""""
def weighted_bin_stats_2d(x, y, z, w, n_bins=50, m_bins=50):
    x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    y_edges = np.linspace(y.min(), y.max(), m_bins + 1)

    wz,    xe, ye, _ = binned_statistic_2d(x, y, z * w, statistic='sum', bins=[x_edges, y_edges])
    w_sum, _,  _,  _ = binned_statistic_2d(x, y, w,     statistic='sum', bins=[x_edges, y_edges])

    wmean = np.full_like(wz, np.nan)
    np.divide(wz, w_sum, out=wmean, where=w_sum > 0)

    bin_x = np.clip(np.digitize(x, x_edges[:-1]) - 1, 0, n_bins - 1)
    bin_y = np.clip(np.digitize(y, y_edges[:-1]) - 1, 0, m_bins - 1)

    bin_mean  = wmean[bin_x, bin_y]
    valid_pts = ~np.isnan(bin_mean)          # ← skip points in empty bins

    residuals = np.zeros_like(z)
    residuals[valid_pts] = (z[valid_pts] - bin_mean[valid_pts]) ** 2

    wres, _, _, _ = binned_statistic_2d(x, y, residuals * w, statistic='sum', bins=[x_edges, y_edges])
    wstd = np.full_like(wres, np.nan)
    np.divide(wres, w_sum, out=wstd, where=w_sum > 0)
    wstd = np.sqrt(wstd) 

    return 0.5*(xe[:-1]+xe[1:]), 0.5*(ye[:-1]+ye[1:]), wmean, wstd

"""Given arrays of RA/Dec (degrees) and a FITS file, return a boolean array of which sources fall within the image footprint."""
def sources_in_fits(ra_deg, dec_deg, fn):
    with fits.open(fn) as hdul:
        w  = WCS(hdul[0].header).celestial
        nx = hdul[0].header["NAXIS1"]
        ny = hdul[0].header["NAXIS2"]

    coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit=u.deg, frame="icrs")
    x, y   = w.world_to_pixel(coords)

    return (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
