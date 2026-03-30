import numpy as np
import matplotlib.pyplot as plt
from functions import match_catalogs_2D, get_combinations, solve_flux_scales
from functions import compute_flux_correction_factor, calculate_correction_factor_weight, calculate_1d_peak, solve_flux_scales_band
from time import perf_counter
from catalog_manager import catalog, config, catalog_set

start = perf_counter()

#### all available catalogs
all_catalogs = catalog_set([
    catalog("/catalogs/racs/racs_gal_clean.fits",             887.5e6,    "racs_gal"),  # the galactic portion of the racs survey
    catalog("/catalogs/racs/racs_full_clean.fits",            887.5e6,    "racs"),      # the rest of the racs survey
    catalog("/catalogs/meerkat/meerkat_clean.fits",           1359.7e6,   "meerkat"),
    catalog("/catalogs/vlssr/vlssr_clean.fits",               73.8e6,     "vlssr"),
    catalog("/catalogs/tgss/tgss_clean.fits",                 150e6,      "tgss"),
    catalog("/catalogs/gleam_300/gleam_300_clean.fits",       300e6,      "gleam_300"),
    catalog("/catalogs/gleam_x_gp/gleam_x_gp_clean.fits",     200e6,      "gleam_xgp"),
    catalog("/catalogs/nvss/nvss_clean.fits",                 1400e6,     "nvss"),
    catalog("/catalogs/wenss/wenss_clean.fits",               325e6,      "wenss"),
    catalog("/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",  144.6e6,    "lofar_dr3"),
    ])

racs_gal, racs, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar_dr3 = all_catalogs.catalogs

#### available configurations
full_config = config(spectral_damping_factor = 5,
                    spectral_index_theory = -0.7,
                    snr_lower_limit = 7,
                    nsigma = 2,
                    minimum_points = 3,
                    crowd_radius_arc = None,
                    minimum_frequency_spacing = None,
                    catalogs = [racs_gal, racs, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar_dr3],
                    reference_file = None,
                    anchor_catalog = lofar_dr3,
                    )

#### Parameters
debug = False
config = full_config
config.setup()

print(f"Setup done at: {perf_counter() - start} s")

##############################################
#### System of equations flux-pair solver ####
##############################################
# cor_matrix = np.zeros((len(config.catalogs), len(config.catalogs)))
# weight_matrix = np.zeros((len(config.catalogs), len(config.catalogs)))

# all_combinations = get_combinations(config.catalogs, size=2)
# output_width = len(str(len(all_combinations)))
# for i, combination in enumerate(all_combinations):
#     local_cats = [config.catalogs[j] for j in combination]
#     output = compute_flux_correction_factor(local_cats, config, debug=False, anchor_override=0)
    
#     if output is not None:
#         spx, snr, cor, flux, catw, max_sep, p_weight, n_crowd, ra, dec = output
#         print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", f"Matches: {len(spx)}")
        
#         tot_wf = calculate_correction_factor_weight(spx, snr, catw, max_sep, p_weight, n_crowd, config)
        
#         filter = tot_wf > 0
#         tot_wf = tot_wf[filter]
#         cor = cor[filter]
        
#         _, _, py = calculate_1d_peak(cor, tot_wf, log=True)
        
#         y, x = combination
#         cor_matrix[x, y] = py
#         cor_matrix[y, x] = 1.0 / py
        
#         weight_matrix[y, x] = np.sum(tot_wf)
#         weight_matrix[x, y] = np.sum(tot_wf)
#     else:
#         print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", "Matches: None")


# scales = solve_flux_scales(cor_matrix, weight_matrix, normalize=True)

# for scale, cat in zip(scales, config.catalogs):
#     print(f"Catalog {cat.name:9} should be multiplied by {scale}")
    

#####################################################################
#### System of equations flux-pair solver declination dependance ####
#####################################################################
bw = 10
decs = np.arange(-90, 90+bw, bw)

cor_matrix = np.zeros((len(config.catalogs), len(config.catalogs), len(decs)))
weight_matrix = np.zeros((len(config.catalogs), len(config.catalogs), len(decs)))

all_combinations = get_combinations(config.catalogs, size=2)
output_width = len(str(len(all_combinations)))
for i, combination in enumerate(all_combinations):
    local_cats = [config.catalogs[j] for j in combination]
    output = compute_flux_correction_factor(local_cats, config, debug=False, anchor_override=0)
    
    if output is not None:
        spx, snr, cor, flux, catw, max_sep, p_weight, n_crowd, ra, dec = output
        print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", f"Matches: {len(spx)}")
        
        for d, declination in enumerate(decs):
            dec_bin = (dec >= declination) & (dec < declination + bw)
            
            tot_wf = calculate_correction_factor_weight(spx[dec_bin], snr[dec_bin], catw[dec_bin], max_sep[dec_bin], p_weight[dec_bin], n_crowd[dec_bin], config)
            
            filter = tot_wf > 0
            tot_wf = tot_wf[filter]
            cor_local = cor[dec_bin][filter]
            
            if len(tot_wf) > 0:
                _, _, py = calculate_1d_peak(cor_local, tot_wf, log=True)
                
                y, x = combination
                cor_matrix[x, y, d] = py
                cor_matrix[y, x, d] = 1.0 / py
                
                weight_matrix[y, x, d] = np.sum(tot_wf)
                weight_matrix[x, y, d] = np.sum(tot_wf)
    else:
        print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", "Matches: None")

scales = []
for d in range(len(decs)):
    s_band = solve_flux_scales_band(cor_matrix[:, :, d], weight_matrix[:, :, d], normalize=True)
    scales.append(s_band)

fig, ax = plt.subplots(figsize=(10, 6))
for i, scale in enumerate(np.array(scales).T):
    ax.plot(decs, scale, label=config.catalogs[i].name)
plt.legend()
plt.xlabel("RA [deg]")
plt.ylabel("Relative correction factor")
plt.show()
