import numpy as np
from astropy.table import Table
import copy
from functions import sources_in_fits, get_pos_err_deg
from pathlib import Path

base_path = Path(__file__).resolve().parent

# wrapper class for incoming Table data
class Catalog:
    def __init__(self, path=None, freq_hz=None, name=None, flux_lim=0, scale=1):
        self.path      = base_path / path.lstrip("/") if path is not None else None
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
    
    def create_subset(self, valid):
        subset = copy.deepcopy(self)
        for attribute in ('flux', 'e_flux', 'ra', 'dec'):
            setattr(subset, attribute, getattr(self, attribute)[valid])
        if self.e_ra  is not None: subset.e_ra  = self.e_ra[valid]
        if self.e_dec is not None: subset.e_dec = self.e_dec[valid]
        if self.err_rad is not None: subset.err_rad = self.err_rad[valid]
        return subset

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
