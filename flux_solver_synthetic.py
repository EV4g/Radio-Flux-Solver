import numpy as np
#import matplotlib.pyplot as plt
from functions import get_combinations, solve_flux_scales, predict_flux
from functions import compute_flux_correction_factor, calculate_correction_factor_weight, calculate_1d_peak#, solve_flux_scales_band
from time import perf_counter
from catalog_manager import Catalog, Config, Catalog_set, Output
from joblib import Parallel, delayed

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

# load a catalog normally, but replace the internal flux data
def inject_synthetic_data(catalogs, ref_flux=1.0, ref_freq=1400e6, spectral_index=-0.8, curvature=0, snr=10):
    for cat in catalogs:
        flux = predict_flux(cat.freq, ref_freq, ref_flux, spectral_index, curvature)
        cat.flux   = np.full(len(cat.ra), flux)
        cat.e_flux = np.full(len(cat.ra), flux / snr)


start = perf_counter()

#### all available catalogs
all_catalogs = Catalog_set([
    Catalog("/catalogs/vlssr/vlssr_clean.fits",              50e6,   "50m",   scale=1),
    Catalog("/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits", 100e6,  "100m",  scale=1),
    Catalog("/catalogs/tgss/tgss_clean.fits",                200e6,  "200m",  scale=1),
    Catalog("/catalogs/gleam_300/gleam_300_clean.fits",      300e6,  "300m",  scale=1),
    Catalog("/catalogs/wenss/wenss_clean.fits",              400e6,  "400m",  scale=1),
    Catalog("/catalogs/racs/racs_low_clean.fits",            500e6,  "500m",  scale=1),
    Catalog("/catalogs/racs/racs_mid_clean.fits",            1000e6, "1000m", scale=1),
    Catalog("/catalogs/racs/racs_high_clean.fits",           1500e6, "1500m", scale=1),
    Catalog("/catalogs/nvss/nvss_clean.fits",                2000e6, "2000m", scale=1),
    ])

#### available configurations
config = Config(spectral_damping_factor = 5,
                spectral_index_theory = -0.8,
                snr_lower_limit = 7,
                nsigma = 2,
                minimum_points = 10,
                crowd_radius_arc = None,
                minimum_frequency_spacing = 0,
                catalogs = all_catalogs.catalogs,
                reference_file = None,
                anchor_catalog = all_catalogs.catalogs[0],
                )

#### Parameters
debug = False
config.setup()

# giving all sets flux values based on a simple powerlaw
inject_synthetic_data(config.catalogs, ref_flux=1.0, ref_freq=1400e6, spectral_index=-0.8, curvature=0, snr=10)

# catalog[1] will be given a flux value twice as high --> result should be a correction factor of 0.5
inject_synthetic_data([config.catalogs[1]], ref_flux=2.0, ref_freq=1400e6, spectral_index=-0.8, curvature=0, snr=10)

# debug print
for cat in config.catalogs: print(f"  {cat.name:>6s}  {cat.freq/1e6:7.1f} MHz  flux={cat.flux[0]:.3f} Jy  e_flux={cat.e_flux[0]:.3f} Jy")

print(f"Setup done at: {(perf_counter() - start):.2f} s")

##############################################
#### System of equations flux-pair solver ####
##############################################
cor_matrix = np.zeros((len(config.catalogs), len(config.catalogs)))
weight_matrix = np.zeros((len(config.catalogs), len(config.catalogs)))

all_combinations = get_combinations(config.catalogs, size=2)
output_width = len(str(len(all_combinations)))

outputs = Parallel(n_jobs=-1)(delayed(compute_flux_correction_factor)([config.catalogs[j] for j in combo], config, debug=False, anchor_override=0) for combo in all_combinations)

for i, (combination, out) in enumerate(zip(all_combinations, outputs)):
    local_cats = [config.catalogs[j] for j in combination]
    
    if out is not None:
        output = Output(*out)
        
        print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", f"Matches: {len(output.ras)}")
        
        tot_wf = calculate_correction_factor_weight(output, config)
        
        filter = tot_wf > 0
        tot_wf = tot_wf[filter]
        output.apply_mask(filter)
        
        _, _, py = calculate_1d_peak(output.correction_factor, tot_wf, log=True)
        
        y, x = combination
        cor_matrix[x, y] = 1.0 / py
        cor_matrix[y, x] = py
        
        weight_matrix[y, x] = np.sum(tot_wf)
        weight_matrix[x, y] = np.sum(tot_wf)
    else:
        print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", "Matches:", colored("None", "yellow"))



scales = solve_flux_scales(cor_matrix, weight_matrix, normalize=True)

print("--------------------------------------------------------------------------")
for i, (scale, cat) in enumerate(zip(scales / scales[config.catalogs.index(config.anchor_catalog)], config.catalogs)):
    print(f"Catalog {cat.name:9} should be multiplied by {scale:.4f}, to get {cat.scale * scale:.4f}", colored("baseline", "yellow") if i == config.catalogs.index(config.anchor_catalog) else "")

print(f"Calculations done at: {(perf_counter() - start):.2f} s")
