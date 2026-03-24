
from scipy.spatial import cKDTree
from scipy.stats import chi2 as _chi2_dist
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

"""Project (RA, Dec) to a local tangent plane centred at (ra0, dec0).
Returns (x, y) in radians; pairwise Euclidean distances in this space are
proper angular separations to < 0.01 mas accuracy for LOFAR-scale fields
(< 10 deg). The key addition over raw (ra,dec) is the cos(dec0) correction
in RA, which removes the dominant first-order error near non-zero declinations."""
def _project_radec(ra_deg, dec_deg, ra0_deg, dec0_deg):
    ra   = np.deg2rad(ra_deg);  dec  = np.deg2rad(dec_deg)
    ra0  = np.deg2rad(ra0_deg); dec0 = np.deg2rad(dec0_deg)
    x = (ra - ra0) * np.cos(dec0)
    y = (dec - dec0)
    return x, y


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------

def match_catalogs_2D(cat_list, thres_arc=2, pos_err_arcsec=None, nsigma=3.0, crowd_radius_arc=None, anchor_index=0, return_quality=False):    
    """Fast n-catalogue cross-matcher with adaptive positional uncertainties,
    crowding detection, and per-pair quality metrics.

    Parameters
    ----------
    cat_list         : list of catalogue objects exposing .ra and .dec (degrees),
                       .flux, and .e_flux.
    thres_arc        : fallback fixed match radius in arcsec. Used only when
                       pos_err_arcsec is None.
    pos_err_arcsec   : per-catalogue 1-D positional RMS in arcsec.  Each entry
                       may be a scalar (applied to all sources in that catalogue)
                       or an array of length len(cat). When given, the match radius
                       for each pair becomes nsigma * sqrt(sigma_a^2 + sigma_b^2),
                       evaluated per-source after an initial coarse query.
    nsigma           : number of combined sigmas used as the acceptance radius
                       when pos_err_arcsec is given (default 3).
    crowd_radius_arc : if given, count same-catalogue neighbours within this
                       radius (arcsec) per source and store in quality['n_crowd'].
                       This implements the crowding / confusion filter discussed
                       in the science review.
    anchor_index     : catalogue index that anchors the multi-catalogue
                       coalescence search (default 0, i.e. LOFAR).
    return_quality   : if True, also return a quality dict (see below).

    Returns  (n == 2)
    -----------------
        idx_0, idx_1  [, quality]

    Returns  (n > 2)
    ----------------
        [idx_0, ..., idx_{n-1}]  [, quality]

    quality dict
    ------------
        'sep_arcsec'  {(a,b): ndarray}  arcsec separation for each matched pair,
                      in the same order as the returned index arrays.
        'p_match'     {(a,b): ndarray}  per-pair match probability from a 2-DOF
                      chi-squared positional test.  Only populated when
                      pos_err_arcsec is given; otherwise an empty dict.
        'n_crowd'     {cat_index: ndarray}  number of same-catalogue neighbours
                      within crowd_radius_arc for each source.  Only populated
                      when crowd_radius_arc is given.
    """
    n = len(cat_list)

    # Build per-catalogue positional-uncertainty arrays  (radians)
    if pos_err_arcsec is not None:
        errs = []
        for i, cat in enumerate(cat_list):
            e = np.asarray(pos_err_arcsec[i], dtype=float)
            if e.ndim == 0 or e.size == 1:
                e = np.full(len(cat.ra), float(e))
            errs.append(np.deg2rad(e / 3600.0))
    else:
        errs = None

    # For each source, counts how many other sources in the same catalogue lie within 
    # crowd_radius_arc arcsec. Sources near others are unreliable
    crowd_counts = {}
    if crowd_radius_arc is not None:
        crowd_r_rad = np.deg2rad(crowd_radius_arc / 3600.0)
        for i, cat in enumerate(cat_list):
            if len(cat.ra) == 0:
                crowd_counts[i] = np.array([], dtype=int)
                continue
            ra0  = np.mean(cat.ra);  dec0 = np.mean(cat.dec)
            cx, cy = _project_radec(cat.ra, cat.dec, ra0, dec0)
            ct   = cKDTree(np.column_stack([cx, cy]))
            nbrs = ct.query_ball_point(np.column_stack([cx, cy]), r=crowd_r_rad)
            crowd_counts[i] = np.array([len(nb) - 1 for nb in nbrs])   # exclude self

    # Pairwise matching
    matched_results = {}   # (a,b) -> (list_idx_a, list_idx_b)
    sep_results     = {}   # (a,b) -> arcsec separations
    prob_results    = {}   # (a,b) -> chi2-based match probabilities
    pair_maps       = {}   # (a,b) -> {idx_b: idx_a}

    for a in range(n):
        for b in range(a + 1, n):
            ra_a, dec_a = np.array(cat_list[a].ra),  np.array(cat_list[a].dec)
            ra_b, dec_b = np.array(cat_list[b].ra),  np.array(cat_list[b].dec)

            # Guard for empty catalogues
            if len(ra_a) == 0 or len(ra_b) == 0:
                matched_results[(a, b)] = ([], [])
                sep_results[(a, b)]     = np.array([])
                prob_results[(a, b)]    = np.array([])
                pair_maps[(a, b)]       = {}
                pair_maps[(b, a)]       = {}
                continue

            if len(ra_a) >= len(ra_b):
                sup_ra, sup_dec, sub_ra, sub_dec, normal = ra_a, dec_a, ra_b, dec_b, True
                err_sup = errs[a] if errs is not None else None
                err_sub = errs[b] if errs is not None else None
            else:
                sup_ra, sup_dec, sub_ra, sub_dec, normal = ra_b, dec_b, ra_a, dec_a, False
                err_sup = errs[b] if errs is not None else None
                err_sub = errs[a] if errs is not None else None

            # tangent-plane projection so that KD-tree distances are more accurate
            ra0  = 0.5 * (np.mean(sup_ra) + np.mean(sub_ra))
            dec0 = 0.5 * (np.mean(sup_dec) + np.mean(sub_dec))
            sup_x, sup_y = _project_radec(sup_ra, sup_dec, ra0, dec0)
            sub_x, sub_y = _project_radec(sub_ra, sub_dec, ra0, dec0)

            # When per-source uncertainties are given, use nsigma * median(combined_sigma) 
            # as the coarse KD-tree radius; otherwise fall back to the fixed thres_arc.
            if errs is not None:
                sigma_pair   = np.median(np.hypot(err_sup, err_sub))
                query_radius = nsigma * sigma_pair
            else:
                query_radius = np.deg2rad(thres_arc / 3600.0)

            tree = cKDTree(np.column_stack([sup_x, sup_y]))
            dists, idxs = tree.query(np.column_stack([sub_x, sub_y]), k=1, distance_upper_bound=query_radius)

            # After the coarse query, recompute the threshold for each candidate pair using its own
            # combined sigma rather than the catalogue-wide median. This is important for SNR-dependent errors
            if errs is not None:
                prelim = dists < query_radius
                accept = np.zeros(len(sub_ra), dtype=bool)
                pidx   = np.where(prelim)[0]
                if len(pidx) > 0:
                    combined_sig    = np.hypot(err_sub[pidx], err_sup[idxs[pidx]])
                    accept[pidx]    = dists[pidx] < nsigma * combined_sig
                valid = accept
            else:
                valid = dists < query_radius

            matched_sub   = np.where(valid)[0]
            matched_sup   = idxs[valid]
            matched_dists = dists[valid]   # radians

            # Keep only the closest sub for each sup
            if len(matched_sup) != len(np.unique(matched_sup)):
                unique_sup, counts = np.unique(matched_sup, return_counts=True)
                dupes = unique_sup[counts > 1]
                keep  = np.ones(len(matched_sub), dtype=bool)
                for dup in dupes:
                    dup_mask       = matched_sup == dup
                    best           = np.argmin(matched_dists[dup_mask])
                    dup_positions  = np.where(dup_mask)[0]
                    keep[dup_positions]       = False
                    keep[dup_positions[best]] = True
                matched_sub   = matched_sub[keep]
                matched_sup   = matched_sup[keep]
                matched_dists = matched_dists[keep]

            # Convert radian KD-tree distances to arcsec separations
            sep_arcsec = np.rad2deg(matched_dists) * 3600.0

            # Per-pair match probability via chi-squared test on the positional separation
            if errs is not None:
                combined_sig_f = np.hypot(err_sub[matched_sub], err_sup[matched_sup])
                chi2_vals = (matched_dists / combined_sig_f) ** 2
                p_match   = 1.0 - _chi2_dist.cdf(chi2_vals, df=2)
            else:
                p_match = np.ones(len(matched_sub))

            # Map back to original a/b catalogue orientations
            if normal:
                idx_a_list = matched_sup.tolist()
                idx_b_list = matched_sub.tolist()
            else:
                idx_a_list = matched_sub.tolist()
                idx_b_list = matched_sup.tolist()

            matched_results[(a, b)] = (idx_a_list, idx_b_list)
            sep_results[(a, b)]     = sep_arcsec
            prob_results[(a, b)]    = p_match

            pair_maps[(a, b)] = dict(zip(idx_b_list, idx_a_list))   # idx_b -> idx_a
            pair_maps[(b, a)] = dict(zip(idx_a_list, idx_b_list))   # idx_a -> idx_b

    # Quality assessment
    quality = {
        'sep_arcsec': sep_results,
        'p_match':    prob_results if errs is not None else {},
        'n_crowd':    crowd_counts,
    }

    # If only two catalogues, already done
    if n == 2:
        i0 = np.array(matched_results[(0, 1)][0])
        i1 = np.array(matched_results[(0, 1)][1])
        if return_quality:
            return i0, i1, quality
        return i0, i1

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


# ===========================================================================
# Test suite
# ===========================================================================

"""Run the full test suite for match_catalogs_2D.
Creates synthetic catalogue objects with known geometry and validates:
  1.  Perfect 2-catalogue match
  2.  Fixed-threshold rejection
  3.  Duplicate / contention handling
  4.  Separation output accuracy
  5.  Adaptive radius via pos_err_arcsec
  6.  Match probability ordering
  7.  Crowding counts
  8.  3-catalogue coalescence
  9.  Edge cases (empty catalogue, single source)
Returns True if all tests pass, False otherwise."""
def test_match_catalogs_2D(verbose=True):

    class _MockCat:
        def __init__(self, ra, dec, flux=None, e_flux=None):
            self.ra     = np.asarray(ra,  dtype=float)
            self.dec    = np.asarray(dec, dtype=float)
            n = len(self.ra)
            self.flux   = np.ones(n)      if flux   is None else np.asarray(flux,   dtype=float)
            self.e_flux = np.full(n, 0.1) if e_flux is None else np.asarray(e_flux, dtype=float)

    results = []

    def _check(name, cond):
        results.append((name, bool(cond)))
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    def _section(title):
        if verbose:
            print(f"\n--- {title} ---")

    if verbose:
        print("=" * 62)
        print("  match_catalogs_2D – test suite")
        print("=" * 62)

    # ----------------------------------------------------------------
    _section("1. Perfect 2-catalogue match")
    ra  = np.array([10.0, 20.0, 30.0])
    dec = np.array([ 0.0,  5.0, -5.0])
    eps = 1e-5   # 0.04 mas, tiny difference
    ca = _MockCat(ra, dec)
    cb = _MockCat(ra + eps, dec + eps)
    i0, i1 = match_catalogs_2D([ca, cb], thres_arc=2)
    _check("All 3 sources recovered",    len(i0) == 3)
    _check("Indices are aligned 1-to-1", np.all(np.sort(i0) == np.sort(i1)))


    # ----------------------------------------------------------------
    _section("2. Fixed-threshold rejection")
    dec2 = dec.copy()
    dec2[1] += 5.0 / 3600.0   # 5 arcsec > 2 arcsecond thresarc
    cb2 = _MockCat(ra, dec2)
    i0, i1 = match_catalogs_2D([ca, cb2], thres_arc=2)
    _check("Only 2 of 3 sources matched", len(i0) == 2)
    _check("Offset source (index 1) excluded", 1 not in i0)


    # ----------------------------------------------------------------
    _section("3. Duplicate / contention handling")
    # Two cat_a sources both within 2" of the same cat_b source; closer one wins
    ra_a3  = np.array([10.0,               10.0 + 0.5/3600.0, 30.0])
    dec_a3 = np.array([ 0.0,               0.0,               0.0])
    ca3    = _MockCat(ra_a3, dec_a3)
    cb3    = _MockCat(np.array([10.0, 30.0]), np.array([0.0, 0.0]))
    i0, i1 = match_catalogs_2D([ca3, cb3], thres_arc=2)
    _check("No duplicate cat_b indices", len(np.unique(i1)) == len(i1))
    _check("At most 2 matches returned", len(i0) <= 2)
    if len(i0) == 2: _check("Closer source (index 0) wins", 0 in i0)


    # ----------------------------------------------------------------
    _section("4. Separation output accuracy (return_quality=True)")
    known_sep = 1.2   # arcsec, in RA at dec=0 so no cos correction needed
    ca4 = _MockCat(np.array([10.0, 20.0]), np.array([0.0, 0.0]))
    cb4 = _MockCat(np.array([10.0, 20.0 + known_sep/3600.0]),
                   np.array([0.0, 0.0]))
    i0, i1, q4 = match_catalogs_2D([ca4, cb4], thres_arc=2, return_quality=True)
    seps = q4['sep_arcsec'][(0, 1)]
    _check("sep_arcsec length == match count", len(seps) == len(i0))
    if len(i0) == 2:
        k0 = int(np.where(i0 == 0)[0][0])
        k1 = int(np.where(i0 == 1)[0][0])
        _check("Near-zero sep for coincident source",      seps[k0] < 0.01)
        _check("Measured sep within 0.1\" of known value", np.abs(seps[k1] - known_sep) < 0.1)


    # ----------------------------------------------------------------
    _section("5. Adaptive radius via pos_err_arcsec")
    # 3 arcsec offset: beyond fixed 2" but within 3 * sqrt(1^2+1^2) = 4.24"
    ca5 = _MockCat(np.array([10.0]), np.array([0.0]))
    cb5 = _MockCat(np.array([10.0 + 3.0/3600.0]), np.array([0.0]))
    i0_fix, _ = match_catalogs_2D([ca5, cb5], thres_arc=2)
    _check("No match with fixed 2\" threshold", len(i0_fix) == 0)
    i0_adp, _ = match_catalogs_2D([ca5, cb5], pos_err_arcsec=[1.0, 1.0], nsigma=3.0)
    _check("Match found with adaptive 3σ threshold", len(i0_adp) == 1)


    # ----------------------------------------------------------------
    _section("6. Match probability (chi-squared test)")
    ca6 = _MockCat(np.array([10.0, 20.0]), np.array([0.0, 0.0]))
    # Source 0: 0.1" offset (p ≈ 0.99); source 1: 2.0" offset (p ≈ 0.02)
    # sigma=0.5" each --> combined=0.707" --> 5σ=3.54" --> both within threshold
    cb6 = _MockCat(np.array([10.0 + 0.1/3600.0, 20.0 + 2.0/3600.0]), np.array([0.0, 0.0]))
    i0, i1, q6 = match_catalogs_2D([ca6, cb6], pos_err_arcsec=[0.5, 0.5], nsigma=5.0, return_quality=True)
    p    = q6['p_match'][(0, 1)]
    seps = q6['sep_arcsec'][(0, 1)]
    _check("p_match populated for both sources", len(p) == 2)
    if len(p) == 2:
        k_close = int(np.argmin(seps))
        k_far   = int(np.argmax(seps))
        _check("Closer match has higher p_match",  p[k_close] > p[k_far])
        _check("Close match p > 0.95",             p[k_close] > 0.95)
        _check("Wide  match p < 0.05",             p[k_far]   < 0.05)


    # ----------------------------------------------------------------
    _section("7. Crowding counts")
    # Sources 0-2 at ~10", ~20" spacing (within 60"); source 3 isolated at ra=50
    ra_c  = np.array([10.0, 10.0 + 10/3600.0, 10.0 + 20/3600.0, 50.0])
    dec_c = np.zeros(4)
    cc    = _MockCat(ra_c, dec_c)
    # thres_arc=0 --> no cross-matching; we only care about crowd_counts here
    _, _, q7 = match_catalogs_2D([cc, cc], thres_arc=0.001,
                                  crowd_radius_arc=60, return_quality=True)
    nc = q7['n_crowd'][0]
    _check("n_crowd length == 4",              len(nc) == 4)
    _check("Crowded sources 0-2 have nc > 0",  np.all(nc[:3] > 0))
    _check("Isolated source 3 has nc == 0",    nc[3] == 0)


    # ----------------------------------------------------------------
    _section("8. 3-catalogue coalescence")
    np.random.seed(0)
    ra_b  = np.random.uniform(10, 20, 50)
    dec_b = np.random.uniform(-5, 5,  50)
    eps3  = 2e-7   # ~0.7 mas
    c1 = _MockCat(ra_b,          dec_b)
    c2 = _MockCat(ra_b + eps3,   dec_b + eps3)
    c3 = _MockCat(ra_b + eps3*2, dec_b)
    res = match_catalogs_2D([c1, c2, c3], thres_arc=2)
    _check("Returns list of 3 arrays",              len(res) == 3)
    _check("All three arrays same length",           len({len(r) for r in res}) == 1)
    _check("≥ 40 sources matched across 3 cats",    len(res[0]) >= 40)
    for ii in range(3):
        _check(f"No duplicate indices in cat {ii}",
               len(np.unique(res[ii])) == len(res[ii]))


    # ----------------------------------------------------------------
    _section("9. Edge cases")
    cat_empty = _MockCat([], [])
    cat_one   = _MockCat([10.0], [0.0])
    i0, i1 = match_catalogs_2D([cat_empty, cat_one], thres_arc=2)
    _check("Empty catalogue → 0 matches",  len(i0) == 0)
    i0, i1 = match_catalogs_2D([cat_one, cat_one], thres_arc=2)
    _check("Single source matches itself", len(i0) == 1 and i0[0] == 0)


    # ----------------------------------------------------------------
    n_pass = sum(r for _, r in results)
    n_fail = len(results) - n_pass
    if verbose:
        print("\n" + "=" * 62)
        print(f"  {n_pass} / {len(results)} tests passed   ({n_fail} failed)")
        print("=" * 62)
    return n_fail == 0
