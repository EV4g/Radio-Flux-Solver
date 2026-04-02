import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy.stats import chi2 as _chi2_dist, binned_statistic_2d
from scipy.stats import chi2

warnings.filterwarnings("ignore", module="matplotlib")

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

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

"""Return beam size [degree] from fits header"""
def get_beam_size(file):
    hdul = fits.open(file)
    header = hdul[0].header
    try:
        return header['BMAJ'], header['BMIN'], header['BPA']
    except:
        return header['CLEANBMJ'], header['CLEANBMN'], header['CLEANBPA']
    return None

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

    # Crowding
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

"""Return indices of catalog of all unique, non-double, <size> combinations with the optional condition f[i] < f[i+1]"""
def get_combinations(cats, size=3, required_index=None, skip_index=None, only_sorted=True, minimum_spacing=0):
    freqs = [cat.freq for cat in cats]
    indexed = sorted(enumerate(zip(freqs, cats)), key=lambda x: x[1][0])
    
    skip     = {skip_index}     if isinstance(skip_index,     int) else set(skip_index)     if skip_index     is not None else set()
    required = {required_index} if isinstance(required_index, int) else set(required_index) if required_index is not None else set()
    
    result = []
    for combo in combinations(indexed, size):
        indices = tuple(i for i, (f, _) in combo)
        freqs_c = [f for _, (f, _) in combo]

        if (freqs_c != sorted(freqs_c)) and only_sorted:        continue
        if np.any(np.diff(sorted(freqs_c)) < minimum_spacing):  continue
        if required and not required.issubset(indices):         continue
        if skip     and any(i in skip for i in indices):        continue

        result.append(indices)
    return result

"""Return indices of catalog of all unique, non-double, <size> permutations with the optional condition f[i] < f[i+1]"""
def get_permutations(cats, size=3, required_index=None, skip_index=None, only_sorted=True, minimum_spacing=0):
    freqs = [cat.freq for cat in cats]
    indexed = sorted(enumerate(zip(freqs, cats)), key=lambda x: x[1][0])
    
    skip     = {skip_index}     if isinstance(skip_index,     int) else set(skip_index)     if skip_index     is not None else set()
    required = {required_index} if isinstance(required_index, int) else set(required_index) if required_index is not None else set()
    
    result = []
    for combo in permutations(indexed, size):
        indices = tuple(i for i, (f, _) in combo)
        freqs_c = [f for _, (f, _) in combo]

        if (freqs_c != sorted(freqs_c)) and only_sorted:        continue
        if np.any(np.diff(sorted(freqs_c)) < minimum_spacing):  continue
        if required and not required.issubset(indices):         continue
        if skip     and any(i in skip for i in indices):        continue

        result.append(indices)
    return result

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

def calculate_1d_peak(x, c, log=False, n=1000):    
    if log: x = np.log10(x)

    mask = np.isfinite(x) & np.isfinite(c) & (c > 0)
    x, c = x[mask], c[mask]

    if len(x) == 0:
        raise ValueError("No finite data points remain after filtering.")

    if x.max() == x.min():
        peak_x0 = 10**x[0] if log else x[0]
        return np.array([peak_x0]), np.array([1.0]), peak_x0
    
    xi = np.linspace(x.min(), x.max(), n)

    H, edges = np.histogram(x, bins=n, weights=c, range=(x.min(), x.max()))
    n_data  = len(x)
    sigma_x = (n_data**(-0.2) * np.std(x)) / ((x.max() - x.min()) / n)
    Zi      = gaussian_filter1d(H, sigma=sigma_x)

    peak_x0 = xi[np.argmax(Zi)]

    kde    = gaussian_kde(x, weights=c)
    result = minimize(lambda p: -kde(p)[0], x0=[peak_x0], method='Nelder-Mead',
                      options={'xatol': 1e-5, 'fatol': 1e-10})
    peak_x0 = result.x[0]

    if log: peak_x0 = 10**peak_x0

    return xi if not log else 10**xi, Zi, peak_x0

def solve_flux_scales(ratio, weight, ref=None, normalize=True):
    """
    ratio[i, j] : pairwise estimate of s_i / s_j
    weight[i,j] : weight for that estimate
    ref         : index of catalog to pin at x_ref = 0 for the solve.
                  If None, 0 is used internally.
    normalize   : if True, rescale s so that geometric mean(s) = 1.

    Returns
    -------
    s : array of relative flux scales, length N.
    """
    N = ratio.shape[0]
    rows = []
    y = []
    w = []

    for i in range(N):
        for j in range(i + 1, N):
            wij = weight[i, j]
            if not np.isfinite(wij) or wij <= 0:
                continue
            rij = ratio[i, j]
            if not np.isfinite(rij) or rij <= 0:
                continue
            rows.append((i, j))
            y.append(np.log(rij))
            w.append(wij)

    y = np.asarray(y)
    w = np.asarray(w)
    m = len(rows)
    if m == 0:
        raise ValueError("No valid pairwise ratios to solve for flux scales")

    A = np.zeros((m, N), dtype=float)
    for k, (i, j) in enumerate(rows):
        A[k, i] = 1.0
        A[k, j] = -1.0

    Wsqrt = np.sqrt(w)
    Aw = A * Wsqrt[:, None]
    yw = y * Wsqrt

    if ref is None:
        ref = 0
    if not (0 <= ref < N):
        raise ValueError(f"ref must be in [0, {N-1}], got {ref}")

    cols_mask = np.arange(N) != ref
    Aw_red = Aw[:, cols_mask]

    x_red, *_ = np.linalg.lstsq(Aw_red, yw, rcond=None)

    x = np.zeros(N, dtype=float)
    x[cols_mask] = x_red
    x[ref] = 0.0

    s = np.exp(x)

    if normalize:
        x0 = np.mean(np.log(s))
        s = np.exp(np.log(s) - x0)

    return s

def solve_flux_scales_band(ratio_slice, weight_slice, normalize=True):
    N = ratio_slice.shape[0]

    # Catalogs with at least one non-zero-weight edge
    active = (weight_slice > 0).any(axis=0)
    idx_active = np.where(active)[0]

    if len(idx_active) == 0:
        return np.full(N, np.nan)

    # Work on the active sub
    ratio_sub  = ratio_slice[np.ix_(idx_active, idx_active)]
    weight_sub = weight_slice[np.ix_(idx_active, idx_active)]

    s_sub = solve_flux_scales(ratio_sub, weight_sub, ref=0, normalize=normalize)

    # NaN for inactive catalogs
    s_full = np.full(N, np.nan)
    s_full[idx_active] = s_sub
    return s_full

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

"""compute the flux correction factor based on three given catalogs. Catalogs are matches, and the last two are used to calculate the spectral index
which is used to extrapolate what the first cat -should- be. The different between -should- and -is-, is the correction factor."""
def compute_flux_correction_factor(cats, config, debug=False, internal_output=False, anchor_override=None, precomputed_indices=None, precomputed_quality=None):
    
    # allow for pre-computed inputs, to skip match_catalogs_2D
    if precomputed_indices is None and precomputed_quality is None:
        indices, quality = match_catalogs_2D(cats, thres_arc=config.thres_arc, return_quality=True, nsigma=config.nsigma, thres_arc_override=config.thres_arc_override, crowd_radius_arc=config.crowd_radius_arc)
    else:
        indices = np.array(precomputed_indices)
        quality = np.array(precomputed_quality)
        
    # if there are too few sources, return None
    if len(indices[0]) <= config.minimum_points:
        if internal_output: print(f"Error: no source-matches found between [{', '.join(f'{cat.name}' for cat in cats)}]")
        return None

    # figure out for which catalog the correction factor should be calculated
    # if cat is given, match that, otherwise assume the first index
    other_index = [i for i in range(len(cats))]
    if config.anchor_catalog is None:
        anchor_index = 0
    else:
        if anchor_override is None:
            try:
                anchor_index = cats.index(config.anchor_catalog)
            except:
                print(colored("Anchor catalog missing from inputs", "light_red"))
                return None
        else:
            anchor_index = anchor_override
    other_index.remove(anchor_index)

    # if crowding parameter is available, store max neighbour (withing crowding_radius) count per source
    n_crowd = np.zeros(len(indices[0]), dtype=int)
    if quality['n_crowd']:
        for cat_idx, match_idx in enumerate(indices):
            nc = quality['n_crowd'].get(cat_idx)
            if nc is not None and len(nc):
                n_crowd = np.maximum(n_crowd, nc[match_idx])

    # create subsets of all catalogs, such that we can ignore (i0,i1,...) afterwards
    cats = [cat.create_subset(indices[index]) for index, cat in enumerate(cats)]

    # all pairwise separations, then take per-source maximum
    coords = [SkyCoord(ra=cat.ra * u.deg, dec=cat.dec * u.deg) for cat in cats]
    sig    = [np.rad2deg(cat.err_rad) * 3600 for cat in cats]   # arcsec per-source

    # pair probability weighting
    pair_seps = {}
    p_weight  = np.ones(len(cats[0].ra))
    n_pairs   = len(cats) * (len(cats) - 1) // 2
    for a, b in combinations(range(len(cats)), 2):
        sep = coords[a].separation(coords[b]).arcsec
        pair_seps[(a, b)] = sep
        p_weight *= 1.0 - chi2.cdf((sep / np.hypot(sig[a], sig[b])) ** 2, df=2)
    p_weight = p_weight ** (1 / n_pairs) # geometric mean of probabilities as normalization
    
    # maximum separation of each matched pair
    max_sep = np.maximum.reduce(list(pair_seps.values()))

    # flux and flux error
    uncorrected_flux = cats[anchor_index].flux
    uncorrected_flux_error = cats[anchor_index].e_flux
    
    # setup a spectral curvature array, will remain theoretical value if not fitted for
    spectral_curvature = np.zeros_like(uncorrected_flux) + config.spectral_curvature_theory
    
    # calculate spectral indices based on available data
    # w.i.p. for case n=4, where curvature can be estimated
    match len(cats):
        case 2:
            spectral_indices = np.ones_like(cats[0].flux) * config.spectral_index_theory
        case 3:
            spectral_indices = get_spectral_index(cats[other_index[0]].flux, cats[other_index[1]].flux, cats[other_index[0]].freq, cats[other_index[1]].freq)
        case 4:
            _, spectral_indices, spectral_curvature = fit_log_parabola([cats[index].freq for index in other_index], [cats[index].flux for index in other_index])
            spectral_indices = spectral_indices + 2 * spectral_curvature * np.log(cats[other_index[0]].freq) # go from log-parabola units to something predict_flux() can use
        case _:
            print(colored(f"Case N={len(cats)} not yet implemented", "light_red"))
            return None

    # fit is based on measuring between two points, linear is average between assuming theoretical extragalactic value for both
    extrapolated_flux_fit = predict_flux(cats[anchor_index].freq, cats[other_index[0]].freq, cats[other_index[0]].flux, spectral_indices, spectral_curvature)
    
    # calculate the linear theoretical flux at anchor_index based on all the other catalogs, and average the result
    extrapolated_flux_linear = np.mean([predict_flux(cats[anchor_index].freq, cats[index].freq, cats[index].flux, config.spectral_index_theory, config.spectral_curvature_theory) for index in other_index], axis=0)
    
    if debug:
        for flux in np.array([cat.flux for cat in cats]).T:
            plt.plot(np.array([cat.freq for cat in cats])*1e-6, flux)
        
        plt.yscale('log')
        plt.ylabel(r"log$_{10}$(Jy)")
        plt.xlabel("Frequency (MHz)")
        plt.show()
        
    # correction factor is the factor to multiply the anchor_catalog flux by to get what it should be, based on the other catalogs
    correction_factor = extrapolated_flux_fit / uncorrected_flux
    snr = uncorrected_flux / uncorrected_flux_error
    
    # calculate average position over all catalogs instead of using the first catalog
    ra = np.average([cat.ra for cat in cats], axis=0)
    dec = np.average([cat.dec for cat in cats], axis=0)
    
    if debug:
        # compare -0.7 assumption versus fitted spectral indices
        mn, mx = min(np.min(extrapolated_flux_fit), np.min(extrapolated_flux_linear)), max(np.max(extrapolated_flux_fit), np.max(extrapolated_flux_linear))
        plt.scatter(extrapolated_flux_linear, extrapolated_flux_fit, c=spectral_indices)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(mn, mx)
        plt.ylim(mn, mx)
        plt.gca().set_box_aspect(1)
        plt.plot((mn, mx), (mn, mx), c='k', ls='--')
        plt.colorbar(label = r"Spectral index $\alpha$")
        plt.xlabel(f"{cats[anchor_index].name} linear flux (Jy)")
        plt.ylabel(f"{cats[anchor_index].name} fitted flux (Jy)")
        plt.title(f"{cats[anchor_index].name} "+r"flux, $\alpha$=-0.7 vs fitted")
        plt.show()
        
        # compare fitted spectral index with correction factor
        plt.scatter(spectral_indices, correction_factor, c=uncorrected_flux, norm='log')
        plt.yscale('log')
        plt.axvline(-0.7, ls='--', c='k')
        plt.axhline(1, ls='--', c='k')
        plt.colorbar(label='Flux (Jy)')
        plt.ylabel("Flux correction factor")
        plt.xlabel(r"Spectral index $\alpha$")
        plt.title("Flux correction as function of spectral index")
        plt.show()
        
    if internal_output: print(f"Completed set [{', '.join(f'{cat.name:9}' for cat in cats)}]", f"Matches: {len(indices[0])}")    
    
    return (spectral_indices, spectral_curvature, snr, correction_factor, extrapolated_flux_fit, max_sep, p_weight, n_crowd, ra, dec)

def fit_log_parabola(freq, flux):
    # ln(flux) = scale + spectral_index ln(nu) + curvature [ln(nu)]^2
    x = np.log(freq)
    y = np.log(flux)
    
    curvature, spectral_index, scale = np.polyfit(x, y, 2) # np.polyfit returns [curvature, spectral_index, scale] for degree=2
    return scale, spectral_index, curvature

def predict_flux_from_parab_fit(freq_target, scale, spectral_index, curvature):
    x = np.log(freq_target)
    log_flux = scale + spectral_index * x + curvature * x**2
    return np.exp(log_flux)

"""Extrapolate flux based on two frequencies, one flux, a spectral index, and an optional curvature parameter"""
def predict_flux(freq_target, freq_reference, flux_reference, spectral_index, curvature=0):
    log_freq_delta = np.log(freq_target / freq_reference)
    log_flux_ratio = spectral_index * log_freq_delta + curvature * log_freq_delta**2
    return flux_reference * np.exp(log_flux_ratio)

"""Calculate weighted correction factor based on per-point spectral indices, signal-to-noise, and correction factor"""
def calculate_correction_factor_weight(spx, snr, max_sep, p_match, n_crowd, config):
    # downweight sources with spectral indices far away from -0.7
    spectral_difference_factor = np.exp(-config.spectral_damping_factor * (spx - config.spectral_index_theory)**2)
    
    # discard low snr sources
    signal_to_noise_factor = snr.copy()
    signal_to_noise_factor[signal_to_noise_factor < config.snr_lower_limit] = 0
    
    # weighting based on separation between points
    # separation_weight = np.exp(-(max_sep / config.thres_arc) ** 2)
    
    return spectral_difference_factor * signal_to_noise_factor * p_match
