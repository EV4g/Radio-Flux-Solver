import numpy as np
from astropy.table import Table
import copy
from functions import sources_in_fits, get_pos_err_deg
from pathlib import Path

base_path = Path(__file__).resolve().parent

# wrapper class for incoming Table data
class catalog:
    def __init__(self, path=None, freq_hz=None, name=None, flux_lim=0, scale=1):
        self.path      = base_path / path.lstrip("/")
        self.freq      = freq_hz    # central frequency
        self.freq_unit = 'Hz'       # frequency unit
        self.name      = name       # survey name
        self.flux_lim  = flux_lim   # lower flux limit; everything below is discarded
        self.scale     = scale      # scale factor, flux data is multiplied by this value
        # data is None until load() is called
        self.flux = self.e_flux = self.flux_unit = None
        self.ra = self.dec = self.e_ra = self.e_dec = None
        self.err_rad = None
    
    def load(self):
        if self.ra is not None: return # already loaded
        
        # read out from disk
        catalog = Table.read(self.path) 
        
        # read out flux data
        self.flux       = np.array(catalog['flux_jy']) * scale
        
        # setup a threshold lower bound based on flux_lim
        flux_threshold = (self.flux > self.flux_lim)
        
        # apply flux_lim threshold
        self.flux       = self.flux[flux_threshold]
        self.e_flux     = np.array(catalog['e_flux_jy'])[flux_threshold] * scale # also apply scale to e_flux
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
    
    def create_subset(self, valid):
        subset = copy.deepcopy(self)
        for attribute in ('flux', 'e_flux', 'ra', 'dec'):
            setattr(subset, attribute, getattr(self, attribute)[valid])
        if self.e_ra  is not None: subset.e_ra  = self.e_ra[valid]
        if self.e_dec is not None: subset.e_dec = self.e_dec[valid]
        if self.err_rad is not None: subset.err_rad = self.err_rad[valid]
        return subset

class catalog_set:
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
class config:
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
                 thres_arc_override            = False):
        
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

            