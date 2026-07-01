import os
import tempfile
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from functions import sources_in_fits, get_pos_err_deg, get_beam_size, radec_to_xyz
from scipy.spatial import cKDTree
from pathlib import Path
import bdsf
from joblib import Parallel, delayed
from time import perf_counter

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

_PROJECT_ROOT = Path(__file__).resolve().parent

def _extract_plane(path):
    """Return a 2-D celestial-plane FITS path; write a temp file only for N-D images."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    if data is None or data.ndim <= 2:
        return path
    plane = data[(0,) * (data.ndim - 2) + (slice(None), slice(None))]
    plane = np.ascontiguousarray(plane.squeeze())
    outhdr = WCS(hdr).celestial.to_header()
    for key in ('BUNIT', 'BMAJ', 'BMIN', 'BPA', 'CLEANBMJ', 'CLEANBMN', 'CLEANBPA'):
        if key in hdr:
            outhdr[key] = hdr[key]
    fd, tmp = tempfile.mkstemp(suffix='.fits', prefix=f'{path.stem}_2d_', dir='/tmp')
    os.close(fd)
    fits.writeto(tmp, plane, outhdr, overwrite=True)
    return Path(tmp)

def _to_icrs(original_path, table):
    """If the original image is Galactic, rotate PyBDSF l/b columns to ICRS in place."""
    with fits.open(original_path) as hdul:
        ctype = WCS(hdul[0].header).celestial.wcs.ctype
    if not any('GLON' in c.upper() or 'GLAT' in c.upper() for c in ctype):
        return
    c = SkyCoord(l=table['ra'], b=table['dec'], unit='deg', frame='galactic').icrs
    table['ra'] = c.ra.deg
    table['dec'] = c.dec.deg

def resolve_catalog_path(path):
    p = Path(path)
    if p.is_absolute() and p.parent.exists():
        return p.resolve()
    clean = str(p).lstrip("/") if p.is_absolute() else str(p)
    return (_PROJECT_ROOT / clean).resolve()

# wrapper class for incoming Table data
class Catalog:
    def __init__(self, path=None, freq_hz=None, name=None, flux_lim=0, scale=1, table=True, reload_cache=True, minimum_position_error=0):
        self.path = resolve_catalog_path(path) if path is not None else None
        if self.path is None: raise ValueError(f"Valid catalog path is required\nPath: {path} is not valid")

        self.dir          = self.path.parent
        self.path_stem    = self.path.stem
        self.freq         = freq_hz       # central frequency
        self.freq_unit    = 'Hz'          # frequency unit
        self.name         = name          # survey name
        self.flux_lim     = flux_lim      # lower flux limit; everything below is discarded
        self.scale        = scale         # scale factor, flux data is multiplied by this value
        self.table        = table         # whether or not the data is a 2D image (False) or table (True)
        self.reload_cache = reload_cache  # whether or not to reload a cached catalog file made from previous image-input
        self.minimum_position_error = minimum_position_error  # arcsec; lower bound for per-source e_ra, e_dec (0 = no floor)
        
        # data is None until load() is called
        self.flux = self.e_flux = self.flux_unit = None
        self.ra = self.dec = self.e_ra = self.e_dec = None
        self.err_rad = None
        
        self._xyz  = None
        self._tree = None
        self._err_rad_median = None  # cached median(err_rad), reused per pair by match_catalogs_2D

    def load(self):
        if self.table:
            if self.ra is not None: return # already loaded
            
            # read out from disk
            catalog = Table.read(self.path) 
            
            # read out flux data
            self.flux       = np.array(catalog['flux_jy']) * self.scale
            
            # setup a threshold lower bound based on flux_lim
            flux_threshold  = (self.flux > self.flux_lim)
            
            # apply flux_lim threshold
            self.flux       = self.flux[flux_threshold]
            self.e_flux     = np.array(catalog['e_flux_jy'])[flux_threshold] * self.scale # also apply scale to e_flux
            self.flux_unit  = str(catalog['flux_jy'].unit)
            
            self.ra         = (np.array(catalog['ra']) % 360)[flux_threshold]
            self.dec        = np.array(catalog['dec'])[flux_threshold]
            
            try:
                self.e_ra   = np.array(catalog['e_ra'])[flux_threshold]
                self.e_dec  = np.array(catalog['e_dec'])[flux_threshold]
                self.e_ra[np.where(np.isnan(self.e_ra))] = 0   # sanitize NaNs
                self.e_dec[np.where(np.isnan(self.e_dec))] = 0 # sanitize NaNs
                if self.minimum_position_error > 0:           # apply lower bound
                    floor_deg = self.minimum_position_error / 3600.0
                    self.e_ra  = np.maximum(self.e_ra,  floor_deg)
                    self.e_dec = np.maximum(self.e_dec, floor_deg)
                self.err_rad = np.deg2rad(get_pos_err_deg(self))
            except KeyError:
                self.e_ra = self.e_dec = self.err_rad = None

        # if not table, then we assume it to be an image
        # we use PyBDSF to still turn it into a catalog
        else:
            image_catalog_path = self.dir / f"{self.path_stem}_catalog.fits"

            if self.reload_cache and image_catalog_path.exists():
                print(colored(f"  Loading cached catalog for {self.path_stem}", "yellow"))
                image_catalog = Table.read(image_catalog_path)
            else:
                print(colored(f"  Running PyBDSF source finding on {self.path_stem}", "yellow"))

                img_path = _extract_plane(self.path)
                try:
                    image = bdsf.process_image(
                        img_path,
                        thresh_isl=3.0,                   # island threshold (sigma)
                        thresh_pix=5.0,                   # peak detection threshold (sigma)
                        rms_box=(200, 50),                # (box_size, step_size) for rms map; tune to your image
                        beam=(get_beam_size(self.path)),  # (maj_deg, min_deg, PA)
                        frequency = self.freq,
                        quiet=True,
                        blank_limit=1e-6,                 # internal mask; values below this (Jy) get ignored
                        outdir='/tmp'
                    )

                    # write pybdsf catalog to file
                    image.write_catalog(outfile=str(image_catalog_path), catalog_type='srl', format='fits', clobber=True)
                    image_catalog = Table.read(image_catalog_path)

                    # stick to convention and overwrite (PyBDSF does not offer column renaming internally)
                    image_catalog.rename_columns(
                        ['RA',  'DEC',  'E_RA', 'E_DEC', 'Total_flux', 'E_Total_flux'],
                        ['ra',  'dec',  'e_ra', 'e_dec', 'flux_jy',    'e_flux_jy']
                    )

                    _to_icrs(self.path, image_catalog)

                    image_catalog.write(image_catalog_path, overwrite=True)

                    print(f"Catalog written to {image_catalog_path}\n")
                finally:
                    if img_path != self.path:
                        img_path.unlink(missing_ok=True)

            # set data based on image data
            self.ra       = np.array(image_catalog['ra'])      # degrees
            self.dec      = np.array(image_catalog['dec'])     # degrees
            self.e_ra     = np.array(image_catalog['e_ra'])    # degrees
            self.e_dec    = np.array(image_catalog['e_dec'])   # degrees
            if self.minimum_position_error > 0:                # apply lower bound
                floor_deg = self.minimum_position_error / 3600.0
                self.e_ra  = np.maximum(self.e_ra,  floor_deg)
                self.e_dec = np.maximum(self.e_dec, floor_deg)
            self.err_rad  = np.deg2rad(get_pos_err_deg(self))
            self.flux     = np.array(image_catalog['flux_jy'])    # Jy (integrated)
            self.e_flux   = np.array(image_catalog['e_flux_jy'])  # Jy
            self.flux_unit = 'Jy'
            
    
    def create_subset(self, valid):
        """Return a new Catalog containing only rows selected by `valid`
        (boolean mask or integer index array. Avoids copy or deepcopy"""
        subset = self.__class__.__new__(self.__class__)
        subset.__dict__.update(self.__dict__)  # shallow copy of every attribute
        
        # Re-bind every per-source array to an independent indexed copy.
        for attr in ('flux', 'e_flux', 'ra', 'dec', 'e_ra', 'e_dec', 'err_rad'):
            v = getattr(self, attr, None)
            if v is not None:
                setattr(subset, attr, v[valid])
        
        # Cached match arrays: __dict__.update copied them by reference, but
        # they're stale (ra/dec just changed). Drop them so match_catalogs_2D
        # rebuilds the tree on first use of this subset. Err-median is also
        # potentially stale (different rows selected).
        subset._xyz             = None
        subset._tree            = None
        subset._err_rad_median  = None
        return subset

    def precompute_match_arrays(self):
        """Build the unit-sphere xyz vectors, KD-tree, and cached err_rad median
        once so match_catalogs_2D can reuse them.

        Tree options balanced_tree=False, compact_nodes=False, make building 
        much faster at the cost of minimal increase in query time.
        """
        if self.ra is None or len(self.ra) == 0:
            self._xyz             = None
            self._tree            = None
            self._err_rad_median  = None
            return
        if self._xyz is None:
            self._xyz = radec_to_xyz(self.ra, self.dec)
        if self._tree is None:
            self._tree = cKDTree(self._xyz, balanced_tree=False, compact_nodes=False)
        if self._err_rad_median is None and self.err_rad is not None:
            self._err_rad_median = float(np.median(self.err_rad))

class Catalog_set:
    registry = {}
    """Registry of catalogs, accessible by name or as an ordered list."""
    def __init__(self, catalogs):
        self._registry = {cat.name: cat for cat in catalogs}
        Catalog_set.registry.update(self._registry)

    def __getattr__(self, name):
        reg = object.__getattribute__(self, '_registry')
        if name in reg:
            return reg[name]
        raise AttributeError(f"No catalog '{name}' in registry")

    def __iter__(self):
        return iter(self._registry.values())

    @property
    def catalogs(self):
        return list(self._registry.values())

def compute_footprint_box(ra_deg, dec_deg, margin_fraction=0.1):
    """Return (ra_min, ra_max, dec_min, dec_max) covering the given positions,
    expanded by margin_fraction of the span on each side. Handles RA wraparound."""
    ra  = np.asarray(ra_deg,  dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    ra_sorted = np.sort(ra)

    # find the largest gap between consecutive RAs (with wraparound at 360°)
    gaps         = np.diff(ra_sorted)
    wrap_gap     = (ra_sorted[0] + 360.0) - ra_sorted[-1]
    largest_gap  = max(gaps.max() if len(gaps) else 0, wrap_gap)
    wraparound   = largest_gap >= 180.0

    if wraparound:
        # bounds run from the end of the largest gap to the start of it
        gap_idx = int(np.argmax(gaps)) if gaps.max() >= wrap_gap else len(ra_sorted) - 1
        ra_min = ra_sorted[(gap_idx + 1) % len(ra_sorted)]
        ra_max = ra_sorted[gap_idx]
    else:
        ra_min, ra_max = float(ra_sorted[0]), float(ra_sorted[-1])

    dec_min, dec_max = float(dec.min()), float(dec.max())

    # expand by margin; convert angular RA margin to degrees via cos(dec)
    ra_span  = (ra_max - ra_min) % 360.0
    dec_span = dec_max - dec_min
    dec_mid  = 0.5 * (dec_min + dec_max)
    ra_margin  = margin_fraction * ra_span  / max(np.cos(np.radians(dec_mid)), 1e-3)
    dec_margin = margin_fraction * dec_span
    if ra_span < 1.0:  # essentially full-sky; keep bounds as [0, 360) so the filter is a no-op
        ra_min, ra_max = 0.0, 360.0
    else:
        ra_min  = (ra_min  - ra_margin) % 360.0
        ra_max  = (ra_max  + ra_margin) % 360.0
    dec_min = max(dec_min - dec_margin, -90.0)
    dec_max = min(dec_max + dec_margin,  90.0)

    return (ra_min, ra_max, dec_min, dec_max)


# wrapper class for passable parameters
class Config:
    def __init__(self, spectral_damping_factor = 5,
                 snr_lower_limit               = 7,
                 spectral_index_theory         = -0.7,
                 minimum_points                = 2,
                 nsigma                        = 3,
                 crowd_radius_arc              = None,
                 minimum_frequency_spacing     = 0,
                 maximum_frequency_spacing     = np.inf,
                 minimum_position_error        = None,
                 catalogs                      = None,
                 catalog_names                 = None,
                 reference_file                = None,
                 footprint_box                 = None,
                 spatial_filter                = False,
                 anchor_catalog                = None,
                 anchor_catalog_name           = None,
                 thres_arc                     = 2,
                 thres_arc_override            = False,
                 spectral_curvature_theory     = 0,
                 higher_order_simple           = False):

        self.thres_arc                 = thres_arc
        self.spectral_damping_factor   = spectral_damping_factor
        self.snr_lower_limit           = snr_lower_limit
        self.minimum_points            = minimum_points
        self.spectral_index_theory     = spectral_index_theory
        self.nsigma                    = nsigma
        self.crowd_radius_arc          = crowd_radius_arc
        self.minimum_frequency_spacing = minimum_frequency_spacing if minimum_frequency_spacing is not None else 0
        self.maximum_frequency_spacing = maximum_frequency_spacing if maximum_frequency_spacing is not None else np.inf
        self.minimum_position_error    = minimum_position_error
        self.higher_order_simple       = higher_order_simple

        if catalogs is not None:
            self.catalogs = list(catalogs)
            self.catalog_names = [cat.name for cat in self.catalogs]
        else:
            self.catalogs = []
            self.catalog_names = list(catalog_names) if catalog_names is not None else []

        if anchor_catalog is not None:
            self.anchor_catalog = anchor_catalog
            self.anchor_catalog_name = anchor_catalog.name
            self.anchor_catalog_index = self.catalogs.index(anchor_catalog) if catalogs is not None else None
        else:
            self.anchor_catalog = None
            self.anchor_catalog_name = anchor_catalog_name
            self.anchor_catalog_index = None

        self.reference_file = resolve_catalog_path(reference_file) if reference_file is not None else None
        self.footprint_box  = footprint_box
        self.spatial_filter = spatial_filter
        self.thres_arc_override         = thres_arc_override
        self.spectral_curvature_theory  = spectral_curvature_theory

    def setup(self):
        # Resolve catalog names to Catalog objects from the global registry
        if self.catalog_names and not self.catalogs:
            self.catalogs = [Catalog_set.registry[name] for name in self.catalog_names]

        # Resolve anchor catalog name to object
        if self.anchor_catalog_name is not None and self.anchor_catalog is None:
            self.anchor_catalog = Catalog_set.registry[self.anchor_catalog_name]

        # Recompute anchor_catalog_index now that self.catalogs is resolved
        if self.anchor_catalog is not None:
            self.anchor_catalog_index = self.catalogs.index(self.anchor_catalog)

            # auto-spatial-filter reference catalogs to the anchor's coverage
            if self.spatial_filter and self.reference_file is None and self.footprint_box is None:
                if self.anchor_catalog.table is False:
                    self.reference_file = str(self.anchor_catalog.path)
                else:
                    self.anchor_catalog.load()
                    self.footprint_box = compute_footprint_box(
                        self.anchor_catalog.ra, self.anchor_catalog.dec, margin_fraction=0.1
                    )
        
        # Propagate minimum_position_error to every catalog
        if self.minimum_position_error is not None:
            for cat in self.catalogs:
                cat.minimum_position_error = self.minimum_position_error

        # load the data per catalog
        for i, cat in enumerate(self.catalogs):
            t0 = perf_counter()
            cat.load()
            n_rows = len(cat.ra) if cat.ra is not None else 0

            # if reference file, remove all points outside of that
            if self.reference_file is not None:
                valid = sources_in_fits(cat.ra, cat.dec, self.reference_file)
                self.catalogs[i] = cat.create_subset(valid)
            elif self.footprint_box is not None:
                ra_min, ra_max, dec_min, dec_max = self.footprint_box
                if ra_max < ra_min:  # box crosses the 0/360° RA seam
                    valid = ((cat.ra >= ra_min) | (cat.ra <= ra_max)) & (cat.dec >= dec_min) & (cat.dec <= dec_max)
                else:
                    valid = ((cat.ra >= ra_min) & (cat.ra <= ra_max) & (cat.dec >= dec_min) & (cat.dec <= dec_max))
                self.catalogs[i] = cat.create_subset(valid)
            if self.reference_file is not None or self.footprint_box is not None:
                if cat is not self.anchor_catalog:
                    print(f"  {cat.name:14s} load+threshold: {(perf_counter()-t0):.2f}s ({n_rows:>8d} rows) --> {len(self.catalogs[i].ra):>8d} rows kept")
            else:
                print(f"  {cat.name:14s} load+threshold: {(perf_counter()-t0):.2f}s ({n_rows:>8d} rows)")
            
            
        # re-bind anchor to the exact same object now sitting in self.catalogs
        if self.anchor_catalog is not None:
            anchor_name = self.anchor_catalog.name
            try:
                self.anchor_catalog = next(c for c in self.catalogs if c.name == anchor_name)
            except StopIteration:
                raise ValueError(f"Anchor_catalog '{anchor_name}' not found in config.catalogs")

        # Parallel precompute of radec_to_xyz and cKDTree
        t0 = perf_counter()
        Parallel(n_jobs=-1, backend='threading')(
            delayed(cat.precompute_match_arrays)() for cat in self.catalogs
        )
        if True:
            print(f"  precompute_match_arrays: {(perf_counter()-t0):.2f}s")

class Output:
    def __init__(self, spx=None, cur=None, snr=None, cor=None, flux=None, sep=None, pmatch=None, ncrowd=None, ra=None, dec=None):
        self.spectral_index     = [] if spx    is None else spx    # per-source spectral index
        self.spectral_curvature = [] if cur    is None else cur    # per-source spectral curvature
        self.signal_to_noise    = [] if snr    is None else snr    # signal-to-noise (flux_jy / e_flux_jy)
        self.correction_factor  = [] if cor    is None else cor    # ratio between read-out anchor_catalog flux and computed flux
        self.fitted_flux        = [] if flux   is None else flux   # anchor_catalog flux based on spectral index extrapolation
        self.max_separation     = [] if sep    is None else sep    # maximum per-source separation between all three matched catalog positions
        self.point_probability  = [] if pmatch is None else pmatch # probability of points matching
        self.crowding_parameter = [] if ncrowd is None else ncrowd # maximum number of neighbours per source within crowd_radius_arc
        self.ras                = [] if ra     is None else ra     # positional coordinates
        self.decs               = [] if dec    is None else dec    # positional coordinates
    
    def add(self, spx, cur, snr, cor, flux, sep, pmatch, ncrowd, ra, dec):
        self.ras.append(ra)
        self.decs.append(dec)
        self.correction_factor.append(cor)
        self.spectral_index.append(spx)
        self.spectral_curvature.append(cur)
        self.fitted_flux.append(flux)
        self.signal_to_noise.append(snr)
        self.max_separation.append(sep)
        self.point_probability.append(pmatch)
        self.crowding_parameter.append(ncrowd)
        
    def concatenate(self):
        self.ras                   = np.concatenate(self.ras)
        self.decs                  = np.concatenate(self.decs)
        self.correction_factor     = np.concatenate(self.correction_factor)
        self.spectral_index        = np.concatenate(self.spectral_index)
        self.spectral_curvature    = np.concatenate(self.spectral_curvature)
        self.fitted_flux           = np.concatenate(self.fitted_flux)
        self.signal_to_noise       = np.concatenate(self.signal_to_noise)
        self.max_separation        = np.concatenate(self.max_separation)
        self.point_probability     = np.concatenate(self.point_probability)
        self.crowding_parameter    = np.concatenate(self.crowding_parameter)
        
    def apply_mask(self, mask):
        self.ras                   = self.ras[mask]
        self.decs                  = self.decs[mask]
        self.correction_factor     = self.correction_factor[mask]
        self.spectral_index        = self.spectral_index[mask]
        self.spectral_curvature    = self.spectral_curvature[mask]
        self.fitted_flux           = self.fitted_flux[mask]
        self.signal_to_noise       = self.signal_to_noise[mask]
        self.max_separation        = self.max_separation[mask]
        self.point_probability     = self.point_probability[mask]
        self.crowding_parameter    = self.crowding_parameter[mask]

    def return_values(self):
        return self.ras, self.decs, self.correction_factor, self.spectral_index, self.spectral_curvature, self.fitted_flux, self.signal_to_noise, self.max_separation, self.point_probability, self.crowding_parameter
