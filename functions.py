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
from scipy.stats import binned_statistic_2d
from scipy.stats import chi2

warnings.filterwarnings("ignore", module="matplotlib")

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

warnings.filterwarnings("ignore", category=FITSFixedWarning)

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
def get_spectral_index(S1, S2, v1, v2, fallback_value=0):
    if v1 != v2:
        return (np.log(S1) - np.log(S2)) / (np.log(v1) - np.log(v2))
    else:
        return fallback_value

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
    except KeyError:
        return header['CLEANBMJ'], header['CLEANBMN'], header['CLEANBPA']
    return None

"""Return a per-source 1D positional RMS (deg) for a catalog.
e_ra / e_dec are combined as sigma_1d = sqrt((e_ra^2 + e_dec^2) / 2)."""
def get_pos_err_deg(cat):
    return np.sqrt((cat.e_ra ** 2 + cat.e_dec ** 2) / 2.0) # degrees

def radec_to_xyz(ra_deg, dec_deg):
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cd  = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])

def angsep_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Per-row angular separation in arcsec using the haversine formula.
    Replacement for SkyCoord.separation(); faster because it skips astropy."""
    r1 = np.deg2rad(ra1_deg);  d1 = np.deg2rad(dec1_deg)
    r2 = np.deg2rad(ra2_deg);  d2 = np.deg2rad(dec2_deg)
    sd = np.sin(0.5 * (d2 - d1))
    sr = np.sin(0.5 * (r2 - r1))
    a  = sd * sd + np.cos(d1) * np.cos(d2) * sr * sr
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(a))) * 3600.0

def match_catalogs_2D(cat_list, thres_arc=2, nsigma=3.0, crowd_radius_arc=None, anchor_index=0, return_quality=False, thres_arc_override=False, workers=-1):
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
    workers            : threads passed to cKDTree.query / query_ball_point. Default -1
                         uses all cores; set to 1 when calling this from a multi-process
                         pool (e.g. joblib.Parallel(n_jobs=-1)) to avoid thread oversubscription.

    Returns
    ----------------
        [idx_0, ..., idx_{n-1}]  [, quality]

    Multi-catalogue (n>=3) result rows form consistent groups: row k of every returned
    array refers to one source in each catalogue that mutually match. Row order is
    anchor-ascending (was insertion-order in older versions; the row-to-row pairing
    is unchanged, only the order across rows differs).

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

    # if thres_arc_override == True, then we ignore the X^2 error matching
    # if any of the err_rad indices are None (improper catalog), also ignore them
    missing_errors = any(cat.err_rad is None for cat in cat_list)
    use_errs = not (thres_arc_override or missing_errors)
    if missing_errors: print(colored("Warning: some catalogs have missing error values, ignoring X^2 error matching", "red"))
    
    # Precompute (or reuse cached) xyz vectors per catalog. When the same catalog is
    # cross-matched repeatedly (e.g. an anchor used across many combos) the caller can
    # set cat._xyz / cat._tree once up-front to avoid rebuilding on every call.
    xyz_all = []
    for cat in cat_list:
        xyz = getattr(cat, '_xyz', None)
        if xyz is None:
            xyz = radec_to_xyz(cat.ra, cat.dec)
        xyz_all.append(xyz)
    errs = [cat.err_rad if use_errs else np.empty(0) for cat in cat_list]
    # Cached medians per catalog (computed once in precompute_match_arrays)
    err_medians = [0.0] * n
    if use_errs:
        for i, cat in enumerate(cat_list):
            m = getattr(cat, '_err_rad_median', None)
            if m is None and errs[i].size:
                m = float(np.median(errs[i]))
                try:
                    cat._err_rad_median = m
                except AttributeError:
                    pass
            if m is not None:
                err_medians[i] = float(m)

    # Prebuild only the KD-trees we actually need: the smaller catalog of each pair is
    # always used as query points (not as a tree), so we skip building its tree unless
    # crowding self-queries require it. Reuses cat._tree if the caller pre-built it.
    sizes = [len(xyz) for xyz in xyz_all]
    need_tree = [(crowd_radius_arc is not None) and s > 0 for s in sizes]
    for a in range(n):
        if sizes[a] == 0: continue
        for b in range(a + 1, n):
            if sizes[b] == 0: continue
            need_tree[a if sizes[a] >= sizes[b] else b] = True
    trees = []
    for i in range(n):
        if not need_tree[i]:
            trees.append(None); continue
        cached = getattr(cat_list[i], '_tree', None)
        trees.append(cached if cached is not None else cKDTree(xyz_all[i]))

    # Crowding (return_length=True returns counts directly — no list-of-lists allocation)
    crowd_counts = {}
    if crowd_radius_arc is not None:
        crowd_r_3d = 2.0 * np.sin(np.deg2rad(crowd_radius_arc / 3600.0) / 2.0)
        for i, (xyz, tree) in enumerate(zip(xyz_all, trees)):
            if tree is None:
                crowd_counts[i] = np.array([], dtype=int)
                continue
            counts = tree.query_ball_point(xyz, r=crowd_r_3d, workers=workers, return_length=True)
            crowd_counts[i] = np.asarray(counts, dtype=int) - 1

    matched_results = {}
    sep_results     = {}
    prob_results    = {}

    for a in range(n):
        for b in range(a + 1, n):
            xyz_a = xyz_all[a]
            xyz_b = xyz_all[b]

            if len(xyz_a) == 0 or len(xyz_b) == 0:
                empty_i = np.empty(0, dtype=np.intp)
                empty_f = np.empty(0)
                matched_results[(a, b)] = (empty_i, empty_i)
                sep_results[(a, b)]     = empty_f
                prob_results[(a, b)]    = empty_f
                continue

            # Always query smaller set against larger tree
            if len(xyz_a) >= len(xyz_b):
                sub_xyz, sup_err, sub_err, tree_sup, normal = xyz_b, errs[a], errs[b], trees[a], True
                sup_med, sub_med = err_medians[a], err_medians[b]
            else:
                sub_xyz, sup_err, sub_err, tree_sup, normal = xyz_a, errs[b], errs[a], trees[b], False
                sup_med, sub_med = err_medians[b], err_medians[a]

            # Coarse search radius (reuse cached medians; avoid redundant median work
            # when the same catalog appears in many pairs)
            if use_errs:
                sigma_pair   = np.hypot(sup_med, sub_med)
                query_radius = 2.0 * np.sin(nsigma * sigma_pair / 2.0)
            else:
                query_radius = 2.0 * np.sin(np.deg2rad(thres_arc / 3600.0) / 2.0)

            dists, idxs = tree_sup.query(sub_xyz, k=1, distance_upper_bound=query_radius, workers=workers)

            # Per-source refinement using actual per-source errors
            prelim = dists < query_radius   # also excludes 'inf' no-match sentinels
            if use_errs:
                pidx = np.flatnonzero(prelim)
                if pidx.size:
                    combined_sig = np.hypot(sub_err[pidx], sup_err[idxs[pidx]])
                    thresh_3d    = 2.0 * np.sin(nsigma * combined_sig / 2.0)
                    valid = np.zeros_like(prelim)
                    valid[pidx] = dists[pidx] < thresh_3d
                else:
                    valid = prelim
            else:
                valid = prelim

            matched_sub   = np.flatnonzero(valid)
            matched_sup   = idxs[valid]
            matched_dists = dists[valid]

            # Deduplication: keep closest sub per sup. Only sort when duplicates actually exist
            # so the natural sub-ascending order is preserved otherwise (matches original output).
            # Sort+diff is ~20x faster than np.unique for the duplicate check.
            n_matched = matched_sup.size
            has_dup = False
            if n_matched > 1:
                _sorted = np.sort(matched_sup)
                has_dup = bool((_sorted[1:] == _sorted[:-1]).any())
            if has_dup:
                order         = np.lexsort((matched_dists, matched_sup))
                matched_sup   = matched_sup[order]
                matched_sub   = matched_sub[order]
                matched_dists = matched_dists[order]
                first         = np.empty(n_matched, dtype=bool)
                first[0]      = True
                np.not_equal(matched_sup[1:], matched_sup[:-1], out=first[1:])
                matched_sup   = matched_sup[first]
                matched_sub   = matched_sub[first]
                matched_dists = matched_dists[first]

            # Convert chord distance to arcsec
            sep_arcsec = np.rad2deg(2.0 * np.arcsin(matched_dists / 2.0)) * 3600.0

            # Match probability: 1 - chi2.cdf(x, df=2) is exactly exp(-x/2). Skip scipy dispatch.
            if use_errs and matched_sub.size:
                sep_rad        = np.deg2rad(sep_arcsec / 3600.0)
                combined_sig_f = np.hypot(sub_err[matched_sub], sup_err[matched_sup])
                chi2_vals      = (sep_rad / combined_sig_f) ** 2
                p_match        = np.exp(-0.5 * chi2_vals)
            elif use_errs:
                p_match = np.empty(0)
            else:
                p_match = np.ones(matched_sub.size)

            if normal:
                idx_a_arr, idx_b_arr = matched_sup, matched_sub
            else:
                idx_a_arr, idx_b_arr = matched_sub, matched_sup

            matched_results[(a, b)] = (idx_a_arr, idx_b_arr)
            sep_results[(a, b)]     = sep_arcsec
            prob_results[(a, b)]    = p_match

    # Quality assessors
    quality = {
        'sep_arcsec': sep_results,
        'p_match':    prob_results if use_errs else {},
        'n_crowd':    crowd_counts,
    }

    # If only two catalogues, already done
    if n == 2:
        i0, i1 = matched_results[(0, 1)]
        if return_quality:
            return (i0, i1), quality
        return (i0, i1)

    # n >= 3: fully vectorized coalescence anchored on anchor_index.
    # After dedup each pair is a 1-to-1 partial bijection, so for each anchor source
    # there is at most one partner per other catalogue. We materialize anchor->other
    # as dense int arrays, then verify cross-pair consistency among non-anchor cats
    # via dense map lookups. No Python-per-source loops, no dicts.
    others = [i for i in range(n) if i != anchor_index]
    sz_anchor = sizes[anchor_index]

    def _pair_indices(a, b):
        """Return (idx_a_in_cat_a, idx_b_in_cat_b) regardless of dict key order."""
        if (a, b) in matched_results:
            return matched_results[(a, b)]
        ib, ia = matched_results[(b, a)]  # stored as (idx_b, idx_a)
        return ia, ib

    if sz_anchor == 0:
        result = [np.empty(0, dtype=np.int64) for _ in range(n)]
    else:
        # partners[k, src] = matched index in cat others[k] for anchor source src, else -1
        partners = np.full((len(others), sz_anchor), -1, dtype=np.int64)
        for k, oc in enumerate(others):
            ia, ib = _pair_indices(anchor_index, oc)
            if ia.size:
                partners[k, ia] = ib

        # Anchor sources that have at least one partner in EVERY other catalogue.
        # Computed via in-place AND to keep peak memory at one bool array.
        valid = partners[0] >= 0
        for k in range(1, len(others)):
            np.logical_and(valid, partners[k] >= 0, out=valid)

        # Cross-pair consistency among non-anchor cats: for each (i, j),
        # m_oi_oj[partners[i]] must equal partners[j] on currently-valid rows.
        # Use the smaller catalogue as the dense-map source to bound peak memory.
        for i in range(len(others)):
            if not valid.any():
                break
            for j in range(i + 1, len(others)):
                if not valid.any():
                    break
                oi, oj = others[i], others[j]
                ia, ib = _pair_indices(oi, oj)
                if ia.size == 0:
                    valid[:] = False
                    break
                if sizes[oi] <= sizes[oj]:
                    m = np.full(sizes[oi], -1, dtype=np.int64)
                    m[ia] = ib
                    # safe gather: partners[i] is guaranteed >= 0 wherever valid is True
                    np.logical_and(valid, m[partners[i]] == partners[j], out=valid)
                else:
                    m = np.full(sizes[oj], -1, dtype=np.int64)
                    m[ib] = ia
                    np.logical_and(valid, m[partners[j]] == partners[i], out=valid)

        accepted = np.flatnonzero(valid)
        result = [None] * n
        result[anchor_index] = accepted
        for k, oc in enumerate(others):
            result[oc] = partners[k, accepted]

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

    # Peak from the smoothed weighted-histogram grid
    peak_idx         = np.argmax(Zi)
    peak_x0, peak_y0 = Xi.ravel()[peak_idx], Yi.ravel()[peak_idx]

    if logx: Xi = 10**Xi; peak_x0 = 10**peak_x0
    if logy: Yi = 10**Yi; peak_y0 = 10**peak_y0

    return Xi, Yi, Zi, peak_x0, peak_y0

def plot_statistics(x, y, weights=None, bins=(50, 50), contour_levels='auto', contour_resolution=1000, percentile_cut=5, show_peaks=True, logx=False, logy=False, xlabel="", ylabel="", title="", save=False):
    # set min max based on percentile cut
    percentile_cut = np.clip(percentile_cut, 0, 50)
    xmin, xmax = np.percentile(x, percentile_cut), np.percentile(x, 100 - percentile_cut)
    ymin, ymax = np.percentile(y, percentile_cut), np.percentile(y, 100 - percentile_cut)
    
    # clip values outside of min max
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    x = x[mask]
    y = y[mask]
    
    # set weights if not given
    weights = weights[mask] if weights is not None else np.ones_like(x)
    
    binx = np.logspace(np.log10(xmin), np.log10(xmax), bins[0]) if logx else np.linspace(xmin, xmax, bins[0])
    biny = np.logspace(np.log10(ymin), np.log10(ymax), bins[1]) if logy else np.linspace(ymin, ymax, bins[1])
    
    # if degenerate arrays, just skip
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return
    
    # if fewer than 50 points, just default to regular scatter plot
    fig, ax = plt.subplots()
    if len(x) < 50: ax.scatter(x, y, c=weights, s=5)
    else:           ax.hist2d(x, y, bins=(binx, biny), weights=weights)
    
    # if auto, use preset, otherwise use given array
    if contour_levels is not None:
        Xi, Yi, Zi, px, py = calculate_contour_statistics(x, y, weights, logy=logy, logx=logx, n=contour_resolution)
        if contour_levels == 'auto': contour_levels= Zi.max() * np.array([0.01, 0.05, 0.15, 0.35, 0.65, 0.90])
        ax.contour(Xi, Yi, Zi, levels=contour_levels, colors='red', alpha=0.7, linewidths=0.8)
    
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    
    # show peak values
    if show_peaks and contour_levels is not None:
        ax.axhline(py, ls="--", color="gray")
        ax.axvline(px, ls="--", color="gray")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save: plt.savefig(f"{title}.png")
    plt.show()

"""A simple way to save fits files to png"""
def fits_to_png(fits_path, output_path, hdu_index=0, vmin=None, vmax=None, cmap="viridis"):
    with fits.open(fits_path) as hdul:
        data = hdul[hdu_index].data[0,0]

    plt.imsave(output_path, data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

"""Return indices of catalog of all unique, non-double, <size> combinations with the optional condition f[i] < f[i+1]"""
def get_combinations(cats, size=3, required_index=None, skip_index=None, only_sorted=True, minimum_spacing=0, maximum_spacing=np.inf):
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
        if np.any(np.diff(sorted(freqs_c)) > maximum_spacing):  continue
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

"""compute the flux correction factor based on three given catalogs. Catalogs are matches, and the last two are used to calculate the spectral index
which is used to extrapolate what the first cat -should- be. The different between -should- and -is-, is the correction factor."""
def compute_flux_correction_factor(cats, config, anchor_override=None, precomputed_indices=None, precomputed_quality=None, workers=-1):
    # allow for pre-computed inputs, to skip match_catalogs_2D
    if precomputed_indices is None and precomputed_quality is None:
        indices, quality = match_catalogs_2D(cats, thres_arc=config.thres_arc, return_quality=True, nsigma=config.nsigma, thres_arc_override=config.thres_arc_override, crowd_radius_arc=config.crowd_radius_arc, workers=workers)
    else:
        indices = np.array(precomputed_indices)
        quality = np.array(precomputed_quality)
        
    # if there are too few sources, return None
    if len(indices[0]) <= config.minimum_points:
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
    sig = [np.rad2deg(cat.err_rad) * 3600 for cat in cats]   # arcsec per-source

    # pair probability weighting. Direct haversine + closed-form chi2 SF (df=2) avoid
    # heavy SkyCoord.separation and scipy.stats dispatch on every matched pair.
    pair_seps = {}
    p_weight  = np.ones(len(cats[0].ra))
    n_pairs   = len(cats) * (len(cats) - 1) // 2
    for a, b in combinations(range(len(cats)), 2):
        sep = angsep_arcsec(cats[a].ra, cats[a].dec, cats[b].ra, cats[b].dec)
        pair_seps[(a, b)] = sep
        chi2_vals = (sep / np.hypot(sig[a], sig[b])) ** 2
        p_weight *= np.exp(-0.5 * chi2_vals)
    p_weight = p_weight ** (1 / n_pairs) # geometric mean of probabilities as normalization
    
    # maximum separation of each matched pair
    max_sep = np.maximum.reduce(list(pair_seps.values()))

    # flux and flux error
    uncorrected_flux = cats[anchor_index].flux
    uncorrected_flux_error = cats[anchor_index].e_flux
    
    # setup spectral index and curvature arrays, will remain theoretical value if not fitted for
    spectral_curvature = np.full_like(uncorrected_flux, config.spectral_curvature_theory)
    spectral_indices   = np.full_like(uncorrected_flux, config.spectral_index_theory)

    # simple case is equal to the N=2 case
    # use theory value and extrapolate. For N>2, average those values per source --> (flux_ref, freq_ref) are each lists
    if config.higher_order_simple or len(cats) == 2:
        flux_ref = [cats[idx].flux for idx in other_index]
        freq_ref = [cats[idx].freq for idx in other_index]
    else:
        # calculate spectral indices based on available data
        match len(cats):
            case 3:
                # use the other two data points to calculate a spectral index
                # extrapolated flux line goes through both ref points, thus (flux_ref, freq_ref) can be single values
                spectral_indices = get_spectral_index(cats[other_index[0]].flux, cats[other_index[1]].flux, cats[other_index[0]].freq, cats[other_index[1]].freq, fallback_value=np.nan)
                flux_ref = [cats[other_index[0]].flux]
                freq_ref = [cats[other_index[0]].freq]
            case 4:
                # use the other three data points to calculate spectral index and spectral curvature
                # # extrapolated flux line goes through all ref points, thus (flux_ref, freq_ref) can be single values
                scale, spectral_indices, spectral_curvature, freq_ref = fit_log_parabola([cats[i].freq for i in other_index], [cats[i].flux for i in other_index])
                flux_ref = [np.exp(scale)]
                freq_ref = [freq_ref]
            case _:
                print(colored(f"Case N={len(cats)} not yet implemented", "light_red"))
                return None

    # fit is based on measuring between two points, linear is average between assuming theoretical extragalactic value for both
    extrapolated_flux_fit = np.average([predict_flux(cats[anchor_index].freq, freq, flux, spectral_indices, spectral_curvature) for (freq, flux) in zip(freq_ref, flux_ref)], axis=0)
    
    # correction factor is the factor to multiply the anchor_catalog flux by to get what it should be, based on the other catalogs
    correction_factor = extrapolated_flux_fit / uncorrected_flux
    snr = uncorrected_flux / uncorrected_flux_error
    
    # calculate average position over all catalogs instead of using the first catalog
    ra = np.average([cat.ra for cat in cats], axis=0)
    dec = np.average([cat.dec for cat in cats], axis=0)
    
    return (spectral_indices, spectral_curvature, snr, correction_factor, extrapolated_flux_fit, max_sep, p_weight, n_crowd, ra, dec)

"""fit log parabola, but now with pivot to ensure scale remains stable"""
def fit_log_parabola(freq, flux, freq_pivot=100e6):
    freq = np.array(freq)
    x = np.log(freq / freq_pivot)
    y = np.log(flux)
    curvature, spectral_index, scale = np.polyfit(x, y, 2)
    return scale, spectral_index, curvature, freq_pivot

"""Extrapolate flux based on two frequencies, one flux, a spectral index, and an optional curvature parameter"""
def predict_flux(freq_target, freq_reference, flux_reference, spectral_index, curvature=0):
    log_freq_delta = np.log(freq_target / freq_reference)
    log_flux_ratio = spectral_index * log_freq_delta + curvature * log_freq_delta**2
    return flux_reference * np.exp(log_flux_ratio)

"""Calculate weighted correction factor based on per-point spectral indices, signal-to-noise, and correction factor"""
def calculate_correction_factor_weight(output, config, sigma_cutoff=6):
    # downweight sources with spectral indices far away from -0.7
    exponent = config.spectral_damping_factor * (output.spectral_index - config.spectral_index_theory)**2
    cutoff = 0.5 * sigma_cutoff**2
    spectral_difference_factor = np.where(exponent < cutoff, np.exp(-exponent), 0.0)
    
    # discard low snr sources
    signal_to_noise_factor = output.signal_to_noise.copy()
    signal_to_noise_factor[signal_to_noise_factor < config.snr_lower_limit] = 0
    
    # weighting based on separation between points
    # separation_weight = np.exp(-(max_sep / config.thres_arc) ** 2)
    
    return spectral_difference_factor * signal_to_noise_factor * output.point_probability

"""Return median of an array with value weights"""
def weighted_median(val, w):
    # searchsorted on the cumulative weight avoids the O(N) bool-mask + indexed-array
    # allocations that the previous "val[np.cumsum(w) >= w.sum()/2][0]" version did.
    idx = np.argsort(val)
    cw  = np.cumsum(w[idx])
    return float(val[idx[np.searchsorted(cw, 0.5 * cw[-1])]])

"""Return biweight location statistic for determining the center of a distribution"""
def biweight_location(arr1, arr2, arr3, weights=None, c=None, max_iter=30, tol=1e-9):
    # stack data
    X = np.column_stack([np.asarray(a, dtype=float) for a in (arr1, arr2, arr3)])

    # load weights, and discard NaN and 0 weights
    w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    keep = np.isfinite(X).all(axis=1) & (w > 0)
    X, w = X[keep], w[keep]

    if len(X) == 0: return (np.nan, np.nan, np.nan)
    
    # normalize weights
    w /= w.sum()
    
    # count the number of non-degenerate axes
    active = np.ptp(X, axis=0) > 0
    
    # calculate auto-rejection threshold
    if c is None: c = np.sqrt(chi2.ppf(0.975, df=max(int(active.sum()), 1)))
    
    # initial guess from simple weighted median
    mu = np.array([weighted_median(X[:, j], w) for j in range(3)])

    # iterative re-weighted least-square loop
    for _ in range(max_iter):
        # median absolute deviation per axis (per-axis loop is more cache-friendly
        # than batching to a (3, N) matrix at large N — the intermediates blow L3)
        mad   = np.array([weighted_median(np.abs(X[:, j] - mu[j]), w) for j in range(3)])
        scale = np.where(mad > 0, mad, 1.0)
        
        # clip to avoid overflow
        diff  = np.clip((X - mu) / scale, -1e7, 1e7)
        dist  = np.sqrt(np.sum(diff[:, active] ** 2, axis=1))
        
        # median joint distance
        med_d = weighted_median(dist, w)
        if med_d == 0: break
        
        # tukey bisquare kernel
        u    = dist / (c * med_d)
        w_bw = np.zeros_like(u)
        w_bw[u < 1] = (1 - u[u < 1] ** 2) ** 2
        
        w_tot = w * w_bw
        if w_tot.sum() == 0: break
        
        mu_new = (w_tot @ X) / w_tot.sum()
        if np.max(np.abs(mu_new - mu)) < tol: mu = mu_new; break
        mu = mu_new
        
    return tuple(mu)
