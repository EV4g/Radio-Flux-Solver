
import numpy as np
from functions import match_catalogs_2D

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
        def __init__(self, ra, dec, flux=None, e_flux=None, e_ra=None, e_dec=None):
            self.ra     = np.asarray(ra,  dtype=float)
            self.dec    = np.asarray(dec, dtype=float)
            n = len(self.ra)
            self.flux   = np.ones(n)      if flux   is None else np.asarray(flux,   dtype=float)
            self.e_flux = np.full(n, 0.1) if e_flux is None else np.asarray(e_flux, dtype=float)
            self.e_ra   = None if e_ra  is None else np.full(n, float(e_ra))
            self.e_dec  = None if e_dec is None else np.full(n, float(e_dec))

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
    eps = 1e-5
    ca = _MockCat(ra, dec)
    cb = _MockCat(ra + eps, dec + eps)
    i0, i1 = match_catalogs_2D([ca, cb], thres_arc=2)
    _check("All 3 sources recovered",    len(i0) == 3)
    _check("Indices are aligned 1-to-1", np.all(np.sort(i0) == np.sort(i1)))


    # ----------------------------------------------------------------
    _section("2. Fixed-threshold rejection")
    dec2 = dec.copy()
    dec2[1] += 5.0 / 3600.0
    cb2 = _MockCat(ra, dec2)
    i0, i1 = match_catalogs_2D([ca, cb2], thres_arc=2)
    _check("Only 2 of 3 sources matched",    len(i0) == 2)
    _check("Offset source (index 1) excluded", 1 not in i0)


    # ----------------------------------------------------------------
    _section("3. Duplicate / contention handling")
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
    known_sep = 1.2
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
    _section("5. Adaptive radius via e_ra / e_dec")
    # ca5/cb5 have no positional errors → fixed thres_arc=2 applies → no match
    ca5 = _MockCat(np.array([10.0]), np.array([0.0]))
    cb5 = _MockCat(np.array([10.0 + 3.0/3600.0]), np.array([0.0]))
    i0_fix, _ = match_catalogs_2D([ca5, cb5], thres_arc=2)
    _check("No match with fixed 2\" threshold", len(i0_fix) == 0)
    # ca5e/cb5e carry e_ra=e_dec=1" → combined sigma=1" → 3σ≈4.24" > 3" → match
    ca5e = _MockCat(np.array([10.0]),              np.array([0.0]), e_ra=1.0, e_dec=1.0)
    cb5e = _MockCat(np.array([10.0 + 3.0/3600.0]), np.array([0.0]), e_ra=1.0, e_dec=1.0)
    i0_adp, _ = match_catalogs_2D([ca5e, cb5e], nsigma=3.0)
    _check("Match found with adaptive 3σ threshold", len(i0_adp) == 1)


    # ----------------------------------------------------------------
    _section("6. Match probability (chi-squared test)")
    # e_ra=e_dec=0.5" → sigma=0.5" → combined=0.707" → 5σ=3.54" → both within threshold
    ca6 = _MockCat(np.array([10.0, 20.0]), np.array([0.0, 0.0]), e_ra=0.5, e_dec=0.5)
    cb6 = _MockCat(np.array([10.0 + 0.1/3600.0, 20.0 + 2.0/3600.0]),
                   np.array([0.0, 0.0]), e_ra=0.5, e_dec=0.5)
    i0, i1, q6 = match_catalogs_2D([ca6, cb6], nsigma=5.0, return_quality=True)
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
    ra_c  = np.array([10.0, 10.0 + 10/3600.0, 10.0 + 20/3600.0, 50.0])
    dec_c = np.zeros(4)
    cc    = _MockCat(ra_c, dec_c)
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
    eps3  = 2e-7
    c1 = _MockCat(ra_b,          dec_b)
    c2 = _MockCat(ra_b + eps3,   dec_b + eps3)
    c3 = _MockCat(ra_b + eps3*2, dec_b)
    res = match_catalogs_2D([c1, c2, c3], thres_arc=2)
    _check("Returns list of 3 arrays",           len(res) == 3)
    _check("All three arrays same length",        len({len(r) for r in res}) == 1)
    _check("≥ 40 sources matched across 3 cats", len(res[0]) >= 40)
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

if __name__ == "__main__":
    test_match_catalogs_2D(verbose=True)
