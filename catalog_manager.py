import numpy as np
from astropy.table import Table
from functions import sources_in_fits, get_pos_err_deg, get_beam_size, radec_to_xyz
from scipy.spatial import cKDTree
from pathlib import Path
import bdsf

base_path = Path(__file__).resolve().parent

# wrapper class for incoming Table data
class Catalog:
    def __init__(self, path=None, freq_hz=None, name=None, flux_lim=0, scale=1, table=True):
        self.path      = base_path / path.lstrip("/") if path is not None else None
        self.dir       = self.path.parent if self.path is not None else None
        self.path_stem = self.path.stem if self.path is not None else None
        self.freq      = freq_hz    # central frequency
        self.freq_unit = 'Hz'       # frequency unit
        self.name      = name       # survey name
        self.flux_lim  = flux_lim   # lower flux limit; everything below is discarded
        self.scale     = scale      # scale factor, flux data is multiplied by this value
        self.table     = table      # whether or not the data is a 2D image (False) or table (True)
        
        # data is None until load() is called
        self.flux = self.e_flux = self.flux_unit = None
        self.ra = self.dec = self.e_ra = self.e_dec = None
        self.err_rad = None
        
        self._xyz  = None
        self._tree = None
    
    def load(self):
        if self.table:
            if self.ra is not None: return # already loaded
            
            # read out from disk
            catalog = Table.read(self.path) 
            
            # read out flux data
            self.flux       = np.array(catalog['flux_jy']) * self.scale
            
            # setup a threshold lower bound based on flux_lim
            flux_threshold = (self.flux > self.flux_lim)
            
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
                self.err_rad = np.deg2rad(get_pos_err_deg(self))
            except Exception:
                self.e_ra = self.e_dec = self.err_rad = None

        # if not table, then we assume it to be an image
        # we use PyBDSF to still turn it into a catalog
        else:
            print(f"Running PyBDSF source finding on {self.path_stem}")
            image = bdsf.process_image(
                self.path,
                thresh_isl=3.0,                   # island threshold (sigma)
                thresh_pix=5.0,                   # peak detection threshold (sigma)
                rms_box=(200, 50),                # (box_size, step_size) for rms map; tune to your image
                beam=(get_beam_size(self.path)),  # (maj_deg, min_deg, PA)
                frequency = self.freq,
                quiet=True,
                blank_limit=1e-6,                 # internal mask; values below this (Jy) get ignored
                outdir='/tmp'
            )
            
            image_catalog_path = f"{self.dir}/{self.path_stem}_catalog.fits"
            image.write_catalog(outfile=image_catalog_path, catalog_type='srl', format='fits', clobber=True)
            image_catalog = Table.read(image_catalog_path)

            # stick to convention and overwrite (PyBDSF does not offer column renaming internally)
            image_catalog.rename_columns(
                ['RA',  'DEC',  'E_RA', 'E_DEC', 'Total_flux', 'E_Total_flux'],
                ['ra',  'dec',  'e_ra', 'e_dec', 'flux_jy',    'e_flux_jy']
            )
            image_catalog.write(image_catalog_path, overwrite=True)

            print(f"Catalog written to {image_catalog_path}\n")
            
            # set data based on image data
            self.ra       = np.array(image_catalog['ra'])      # degrees
            self.dec      = np.array(image_catalog['dec'])     # degrees
            self.e_ra     = np.array(image_catalog['e_ra'])    # degrees
            self.e_dec    = np.array(image_catalog['e_dec'])   # degrees
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
        # rebuilds the tree on first use of this subset.
        subset._xyz  = None
        subset._tree = None
        return subset
    
    def precompute_match_arrays(self):
        """Build the unit-sphere xyz vectors and KD-tree once so match_catalogs_2D
        can reuse them across many calls. Should be called after load() on every catalog 
        that will participate in cross-matching."""
        if self.ra is None or len(self.ra) == 0:
            self._xyz  = None
            self._tree = None
            return
        if self._xyz is None:
            self._xyz = radec_to_xyz(self.ra, self.dec)
        if self._tree is None:
            self._tree = cKDTree(self._xyz)

class Catalog_set:
    """Registry of catalogs, accessible by name or as an ordered list."""
    def __init__(self, catalogs):
        self._registry = {cat.name: cat for cat in catalogs}

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

# wrapper class for passable parameters
class Config:
    def __init__(self, spectral_damping_factor = 5,
                 snr_lower_limit               = 7,
                 spectral_index_theory         = -0.7,
                 minimum_points                = 2,
                 nsigma                        = 3,
                 crowd_radius_arc              = None,
                 minimum_frequency_spacing     = None,
                 catalogs                      = None,
                 reference_file                = None,
                 anchor_catalog                = None,
                 thres_arc                     = 2,
                 thres_arc_override            = False,
                 spectral_curvature_theory     = 0):
        
        self.thres_arc                  = thres_arc
        self.spectral_damping_factor    = spectral_damping_factor
        self.snr_lower_limit            = snr_lower_limit
        self.minimum_points             = minimum_points
        self.spectral_index_theory      = spectral_index_theory
        self.nsigma                     = nsigma
        self.crowd_radius_arc           = crowd_radius_arc
        self.minimum_frequency_spacing  = minimum_frequency_spacing
        self.catalogs                   = list(catalogs) if catalogs is not None else []
        self.catalog_names              = [cat.name for cat in self.catalogs] if catalogs is not None else []
        self.reference_file             = reference_file
        self.anchor_catalog             = anchor_catalog
        self.anchor_catalog_index       = self.catalogs.index(anchor_catalog) if catalogs is not None else None
        self.thres_arc_override         = thres_arc_override
        self.spectral_curvature_theory  = spectral_curvature_theory
        
    def setup(self):
        # load the data per catalog
        for i, cat in enumerate(self.catalogs):
            cat.load()
            
            # if reference file, remove all points outside of that
            if self.reference_file is not None:
                valid = sources_in_fits(cat.ra, cat.dec, self.reference_file)
                self.catalogs[i] = cat.create_subset(valid)

        # re-bind anchor to the exact same object now sitting in self.catalogs
        if self.anchor_catalog is not None:
            anchor_name = self.anchor_catalog.name
            try:
                self.anchor_catalog = next(c for c in self.catalogs if c.name == anchor_name)
            except StopIteration:
                raise ValueError(f"Anchor_catalog '{anchor_name}' not found in config.catalogs")
        
        # Precompute xyz/tree for each catalog
        for cat in self.catalogs:
            cat.precompute_match_arrays()

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
