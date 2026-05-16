# Test suite for match_catalogs_2D and the cross-match pipeline.

import numpy as np
from scipy.spatial import cKDTree

from functions import match_catalogs_2D, radec_to_xyz, angsep_arcsec


# ---------------------------------------------------------------------------
# Mock catalog: matches the attribute surface that match_catalogs_2D consumes
# (ra, dec, err_rad, _xyz, _tree, _err_rad_median).
# ---------------------------------------------------------------------------
class _MockCat:
    def __init__(self, ra, dec, flux=None, e_flux=None,
                 e_ra_arcsec=None, e_dec_arcsec=None, name="mock"):
        self.name   = name
        self.ra     = np.asarray(ra,  dtype=float)
        self.dec    = np.asarray(dec, dtype=float)
        n = len(self.ra)
        self.flux   = np.ones(n)      if flux   is None else np.asarray(flux,   dtype=float)
        self.e_flux = np.full(n, 0.1) if e_flux is None else np.asarray(e_flux, dtype=float)
        self.freq   = 1.0e9

        # Per-source 1-D positional sigma (radians). Mirrors Catalog.load():
        # combines e_ra and e_dec as sigma = sqrt((e_ra^2 + e_dec^2)/2).
        if e_ra_arcsec is None and e_dec_arcsec is None:
            self.e_ra = self.e_dec = self.err_rad = None
        else:
            era_deg = (e_ra_arcsec  if e_ra_arcsec  is not None else 0.0) / 3600.0
            edc_deg = (e_dec_arcsec if e_dec_arcsec is not None else 0.0) / 3600.0
            self.e_ra  = np.full(n, era_deg)
            self.e_dec = np.full(n, edc_deg)
            self.err_rad = np.deg2rad(np.sqrt((self.e_ra**2 + self.e_dec**2) / 2.0))

        # Cache slots used by match_catalogs_2D — start empty; the function will
        # build them lazily, and we can also pre-populate them in specific tests.
        self._xyz             = None
        self._tree            = None
        self._err_rad_median  = None


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_match_catalogs_2D(verbose=True):
    results = []

    def _check(name, cond):
        results.append((name, bool(cond)))
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    def _section(title):
        if verbose:
            print(f"\n--- {title} ---")

    if verbose:
        print("=" * 70)
        print("  match_catalogs_2D – test suite")
        print("=" * 70)

    # ----------------------------------------------------------------
    _section("1. Perfect 2-catalogue match (fixed threshold)")
    ra  = np.array([10.0, 20.0, 30.0])
    dec = np.array([ 0.0,  5.0, -5.0])
    eps = 1e-5  # ~0.04 arcsec offset
    ca = _MockCat(ra, dec)
    cb = _MockCat(ra + eps, dec + eps)
    i0, i1 = match_catalogs_2D([ca, cb], thres_arc=2, thres_arc_override=True)
    _check("All 3 sources recovered",     len(i0) == 3)
    _check("Per-row pairing aligned",     np.array_equal(np.sort(i0), np.sort(i1)))
    _check("Returns ndarray of ints",     isinstance(i0, np.ndarray) and np.issubdtype(i0.dtype, np.integer))

    # ----------------------------------------------------------------
    _section("2. Fixed-threshold rejection")
    dec2 = dec.copy(); dec2[1] += 5.0 / 3600.0   # shift source 1 by 5 arcsec
    cb2 = _MockCat(ra, dec2)
    i0, i1 = match_catalogs_2D([ca, cb2], thres_arc=2, thres_arc_override=True)
    _check("Only 2 of 3 sources matched",       len(i0) == 2)
    _check("Offset source (index 1) excluded",  1 not in i0)

    # ----------------------------------------------------------------
    _section("3. Duplicate / contention (closest wins)")
    ra_a3  = np.array([10.0,               10.0 + 0.5/3600.0, 30.0])
    dec_a3 = np.array([ 0.0,               0.0,               0.0])
    ca3    = _MockCat(ra_a3, dec_a3)
    cb3    = _MockCat(np.array([10.0, 30.0]), np.array([0.0, 0.0]))
    i0, i1 = match_catalogs_2D([ca3, cb3], thres_arc=2, thres_arc_override=True)
    _check("No duplicate cat_b indices",   len(np.unique(i1)) == len(i1))
    _check("At most 2 matches returned",   len(i0) <= 2)
    if len(i0) == 2:
        _check("Closer source (index 0) wins", 0 in i0)

    # ----------------------------------------------------------------
    _section("4. Separation output accuracy (return_quality=True)")
    known_sep = 1.2   # arcsec
    ca4 = _MockCat(np.array([10.0, 20.0]), np.array([0.0, 0.0]))
    cb4 = _MockCat(np.array([10.0, 20.0 + known_sep/3600.0]), np.array([0.0, 0.0]))
    (i0, i1), q4 = match_catalogs_2D([ca4, cb4], thres_arc=2, thres_arc_override=True,
                                     return_quality=True)
    seps = q4['sep_arcsec'][(0, 1)]
    _check("quality has sep_arcsec key",          (0, 1) in q4['sep_arcsec'])
    _check("sep_arcsec length == match count",    len(seps) == len(i0))
    if len(i0) == 2:
        k0 = int(np.where(i0 == 0)[0][0])
        k1 = int(np.where(i0 == 1)[0][0])
        _check("Near-zero sep for coincident source",        seps[k0] < 0.01)
        _check("Measured sep within 0.1\" of known value",   abs(seps[k1] - known_sep) < 0.1)

    # ----------------------------------------------------------------
    _section("5. Adaptive radius via err_rad (use_errs path)")
    # No errs => use thres_arc_override-style fixed 2" threshold => source 3" apart rejected
    ca5 = _MockCat(np.array([10.0]), np.array([0.0]))
    cb5 = _MockCat(np.array([10.0 + 3.0/3600.0]), np.array([0.0]))
    i0_fix, _ = match_catalogs_2D([ca5, cb5], thres_arc=2, thres_arc_override=True)
    _check("No match with fixed 2\" threshold", len(i0_fix) == 0)
    # With err_rad=1" per source, combined sigma ~1.0", 3-sigma ~ 3" → just barely accepts 3"
    ca5e = _MockCat(np.array([10.0]),              np.array([0.0]),
                    e_ra_arcsec=1.0, e_dec_arcsec=1.0)
    cb5e = _MockCat(np.array([10.0 + 3.0/3600.0]), np.array([0.0]),
                    e_ra_arcsec=1.0, e_dec_arcsec=1.0)
    i0_adp, _ = match_catalogs_2D([ca5e, cb5e], nsigma=3.5)
    _check("Match found with adaptive 3.5σ threshold", len(i0_adp) == 1)

    # ----------------------------------------------------------------
    _section("6. p_match orders by separation (chi-squared SF)")
    ca6 = _MockCat(np.array([10.0, 20.0]), np.array([0.0, 0.0]),
                   e_ra_arcsec=0.5, e_dec_arcsec=0.5)
    cb6 = _MockCat(np.array([10.0 + 0.1/3600.0, 20.0 + 2.0/3600.0]),
                   np.array([0.0, 0.0]), e_ra_arcsec=0.5, e_dec_arcsec=0.5)
    (i0, i1), q6 = match_catalogs_2D([ca6, cb6], nsigma=5.0, return_quality=True)
    p    = q6['p_match'][(0, 1)]
    seps = q6['sep_arcsec'][(0, 1)]
    _check("p_match populated", len(p) == 2)
    if len(p) == 2:
        k_close = int(np.argmin(seps))
        k_far   = int(np.argmax(seps))
        _check("Closer match has higher p_match", p[k_close] > p[k_far])
        _check("All p_match in [0, 1]",            np.all((p >= 0) & (p <= 1)))
        # Closed-form check: p = exp(-chi2/2). At sep=0 we expect p≈1.
        _check("Close match p > 0.95",             p[k_close] > 0.95)

    # ----------------------------------------------------------------
    _section("7. Crowding counts (self-match)")
    ra_c  = np.array([10.0, 10.0 + 10/3600.0, 10.0 + 20/3600.0, 50.0])
    dec_c = np.zeros(4)
    cc    = _MockCat(ra_c, dec_c)
    (_, _), q7 = match_catalogs_2D([cc, cc], thres_arc=0.001, thres_arc_override=True,
                                   crowd_radius_arc=60, return_quality=True)
    nc = q7['n_crowd'][0]
    _check("n_crowd length == 4",             len(nc) == 4)
    _check("Crowded sources 0-2 have nc > 0", np.all(nc[:3] > 0))
    _check("Isolated source 3 has nc == 0",   nc[3] == 0)

    # ----------------------------------------------------------------
    _section("8. 3-catalogue coalescence (vectorized n>=3 path)")
    rng = np.random.default_rng(0)
    ra_b  = rng.uniform(10, 20, 50)
    dec_b = rng.uniform(-5,  5, 50)
    eps3  = 2e-7  # tiny offset to guarantee match within any threshold
    c1 = _MockCat(ra_b,            dec_b)
    c2 = _MockCat(ra_b + eps3,     dec_b + eps3)
    c3 = _MockCat(ra_b + eps3 * 2, dec_b)
    res = match_catalogs_2D([c1, c2, c3], thres_arc=2, thres_arc_override=True)
    _check("Returns list of 3 arrays",            len(res) == 3)
    _check("All three arrays same length",        len({len(r) for r in res}) == 1)
    _check("≥ 40 sources matched across 3 cats",  len(res[0]) >= 40)
    for ii in range(3):
        _check(f"No duplicate indices in cat {ii}",
               len(np.unique(res[ii])) == len(res[ii]))
    # Per-row group consistency: every matched triple is pairwise within threshold.
    # Aggregate into a single check (previous per-row check was registering only on
    # failure, leaving the success path uncounted).
    cat_ras  = [c1.ra, c2.ra, c3.ra]
    cat_decs = [c1.dec, c2.dec, c3.dec]
    rows_ok = 0
    for k in range(len(res[0])):
        ai_pairs = [(0, 1), (0, 2), (1, 2)]
        pair_ok = True
        for ai, bi in ai_pairs:
            i_a = int(res[ai][k]);  i_b = int(res[bi][k])
            d = float(angsep_arcsec(
                np.array([cat_ras[ai][i_a]]),  np.array([cat_decs[ai][i_a]]),
                np.array([cat_ras[bi][i_b]]),  np.array([cat_decs[bi][i_b]]),
            )[0])
            if d > 2.0:
                pair_ok = False
                break
        if pair_ok:
            rows_ok += 1
    _check("Every matched triple is pairwise within 2\"", rows_ok == len(res[0]))

    # ----------------------------------------------------------------
    _section("9. 4-catalogue coalescence")
    c4 = _MockCat(ra_b + eps3 * 3, dec_b + eps3)
    res4 = match_catalogs_2D([c1, c2, c3, c4], thres_arc=2, thres_arc_override=True,
                             anchor_index=0)
    _check("Returns list of 4 arrays",           len(res4) == 4)
    _check("All four arrays same length",        len({len(r) for r in res4}) == 1)
    _check("≥ 40 sources matched across 4 cats", len(res4[0]) >= 40)

    # ----------------------------------------------------------------
    _section("10. Edge cases (empty / single-source)")
    cat_empty = _MockCat([], [])
    cat_one   = _MockCat([10.0], [0.0])
    i0, i1 = match_catalogs_2D([cat_empty, cat_one], thres_arc=2, thres_arc_override=True)
    _check("Empty + single → 0 matches",  len(i0) == 0)
    i0, i1 = match_catalogs_2D([cat_one, cat_one], thres_arc=2, thres_arc_override=True)
    _check("Single matches itself",        len(i0) == 1 and i0[0] == 0)
    # n=3 with one empty catalog
    res = match_catalogs_2D([cat_one, cat_one, cat_empty], thres_arc=2, thres_arc_override=True)
    _check("n=3 with one empty cat → all empty results",
           all(len(r) == 0 for r in res))

    # ----------------------------------------------------------------
    _section("11. Cached _xyz / _tree reuse (perf path)")
    ra_p  = rng.uniform(10, 20, 200)
    dec_p = rng.uniform(-5,  5, 200)
    cp1 = _MockCat(ra_p,           dec_p)
    cp2 = _MockCat(ra_p + eps3,    dec_p)
    # Manually populate the cache and ensure match_catalogs_2D uses it (we plant
    # a "wrong" xyz that should yield the same shape but trip a check)
    cp1._xyz  = radec_to_xyz(cp1.ra, cp1.dec)
    cp1._tree = cKDTree(cp1._xyz)
    cp2._xyz  = radec_to_xyz(cp2.ra, cp2.dec)
    cp2._tree = cKDTree(cp2._xyz)
    pre_tree = cp1._tree
    i0, i1 = match_catalogs_2D([cp1, cp2], thres_arc=2, thres_arc_override=True)
    _check("Cached tree not replaced",     cp1._tree is pre_tree)
    _check("Match count matches (200)",    len(i0) == 200)

    # ----------------------------------------------------------------
    _section("12. Cached _err_rad_median is used (no extra median calls)")
    ca12 = _MockCat(np.array([10.0, 20.0, 30.0]), np.zeros(3), e_ra_arcsec=0.5, e_dec_arcsec=0.5)
    cb12 = _MockCat(np.array([10.0, 20.0, 30.0]) + 0.2/3600, np.zeros(3), e_ra_arcsec=0.5, e_dec_arcsec=0.5)
    assert ca12.err_rad is not None and cb12.err_rad is not None
    ca12._err_rad_median = float(np.median(ca12.err_rad))
    cb12._err_rad_median = float(np.median(cb12.err_rad))
    pre_med_a = ca12._err_rad_median
    pre_med_b = cb12._err_rad_median
    i0, i1 = match_catalogs_2D([ca12, cb12], nsigma=3.0)
    _check("Cached err_rad_median unchanged on a",  ca12._err_rad_median == pre_med_a)
    _check("Cached err_rad_median unchanged on b",  cb12._err_rad_median == pre_med_b)
    _check("Match count == 3",                       len(i0) == 3)
    # Lazy fill: a cat with no cache should get _err_rad_median populated
    ca12b = _MockCat(np.array([10.0]), np.zeros(1), e_ra_arcsec=0.5, e_dec_arcsec=0.5)
    cb12b = _MockCat(np.array([10.0]), np.zeros(1), e_ra_arcsec=0.5, e_dec_arcsec=0.5)
    _ = match_catalogs_2D([ca12b, cb12b], nsigma=3.0)
    _check("Lazy fill: _err_rad_median populated", ca12b._err_rad_median is not None)

    # ----------------------------------------------------------------
    _section("13. workers parameter passthrough")
    # Just ensure both workers=-1 (default) and workers=1 produce identical results
    rng = np.random.default_rng(13)
    ra13  = rng.uniform(10, 20, 500)
    dec13 = rng.uniform(-5, 5, 500)
    ca13 = _MockCat(ra13, dec13)
    cb13 = _MockCat(ra13 + eps3, dec13)
    i0_a, i1_a = match_catalogs_2D([ca13, cb13], thres_arc=2, thres_arc_override=True, workers=-1)
    i0_b, i1_b = match_catalogs_2D([ca13, cb13], thres_arc=2, thres_arc_override=True, workers=1)
    _check("workers=-1 vs workers=1 identical i0", np.array_equal(i0_a, i0_b))
    _check("workers=-1 vs workers=1 identical i1", np.array_equal(i1_a, i1_b))

    # ----------------------------------------------------------------
    _section("14. Catalog.create_subset isolation (regression guard)")
    # Only run if catalog_manager imports cleanly without bdsf side-effects
    try:
        from catalog_manager import Catalog
        cat = Catalog.__new__(Catalog)
        cat.path = cat.dir = cat.path_stem = None
        cat.freq = 1e9; cat.freq_unit = 'Hz'; cat.name = 't'; cat.flux_lim = 0
        cat.scale = 1; cat.table = True; cat.flux_unit = 'Jy'
        cat._xyz = cat._tree = cat._err_rad_median = None
        N = 8
        cat.flux    = np.arange(N, dtype=float)
        cat.e_flux  = np.full(N, 0.1)
        cat.ra      = np.linspace(10, 20, N)
        cat.dec     = np.linspace(-5, 5, N)
        cat.e_ra    = np.full(N, 0.001)
        cat.e_dec   = np.full(N, 0.001)
        cat.err_rad = np.full(N, 1e-6)
        cat.precompute_match_arrays()
        # confirm precompute populated all three caches
        _check("precompute fills _xyz",            cat._xyz is not None)
        _check("precompute fills _tree",           cat._tree is not None)
        _check("precompute fills _err_rad_median", cat._err_rad_median is not None)

        mask = np.array([True, False, True, True, False, True, False, True])
        sub  = cat.create_subset(mask)
        _check("subset has expected size",         len(sub.ra) == int(mask.sum()))
        _check("subset.ra independent of cat.ra",  not np.shares_memory(cat.ra, sub.ra))
        _check("subset.flux owndata",              sub.flux.flags.owndata)
        sub.ra[:] = 999.0
        _check("Mutating subset leaves cat.ra intact",
               np.allclose(cat.ra, np.linspace(10, 20, N)))
        _check("subset cache invalidated (_xyz)",   sub._xyz  is None)
        _check("subset cache invalidated (_tree)",  sub._tree is None)
        _check("subset cache invalidated (_err_rad_median)",
               sub._err_rad_median is None)
    except Exception as e:
        _check(f"catalog_manager import/test failed: {type(e).__name__}: {e}", False)

    # ----------------------------------------------------------------
    n_pass = sum(r for _, r in results)
    n_fail = len(results) - n_pass
    if verbose:
        print("\n" + "=" * 70)
        print(f"  {n_pass} / {len(results)} tests passed   ({n_fail} failed)")
        print("=" * 70)
    return n_fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if test_match_catalogs_2D(verbose=True) else 1)
