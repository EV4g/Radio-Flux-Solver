import numpy as np
#import matplotlib.pyplot as plt
from functions import get_combinations, solve_flux_scales#, calculate_contour_statistics
from functions import compute_flux_correction_factor, calculate_correction_factor_weight, calculate_1d_peak#, solve_flux_scales_band
from time import perf_counter
from catalog_manager import Catalog, Config, Catalog_set, Output
from joblib import Parallel, delayed

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

start = perf_counter()

#### all available catalogs
all_catalogs = Catalog_set([
    Catalog("/catalogs/racs/racs_low_gal_clean.fits",         887.5e6,    "racs_gal",   scale=1),  # the galactic portion of the racs-low survey
    Catalog("/catalogs/racs/racs_low_clean.fits",             887.5e6,    "racs_low",   scale=1),  # the rest of the racs-low survey
    Catalog("/catalogs/racs/racs_mid_clean.fits",             1367.5e6,   "racs_mid",   scale=1),
    Catalog("/catalogs/racs/racs_high_clean.fits",            1655.5e6,   "racs_high",  scale=1),
    Catalog("/catalogs/meerkat/meerkat_clean.fits",           1359.7e6,   "meerkat",    scale=1),
    Catalog("/catalogs/vlssr/vlssr_clean.fits",               73.8e6,     "vlssr",      scale=1),
    Catalog("/catalogs/tgss/tgss_clean.fits",                 150e6,      "tgss",       scale=1),
    Catalog("/catalogs/gleam_300/gleam_300_clean.fits",       300e6,      "gleam_300",  scale=1),
    Catalog("/catalogs/gleam_x_gp/gleam_x_gp_clean.fits",     200e6,      "gleam_xgp",  scale=1),
    Catalog("/catalogs/nvss/nvss_clean.fits",                 1400e6,     "nvss",       scale=1),
    Catalog("/catalogs/wenss/wenss_clean.fits",               325e6,      "wenss",      scale=1),
    Catalog("/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",  144.6e6,    "lofar_dr3",  scale=1),
    ])

racs_gal, racs_low, racs_mid, racs_high, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar_dr3 = all_catalogs.catalogs

#### available configurations
full_config = Config(spectral_damping_factor = 5,
                    spectral_index_theory = -0.8,
                    snr_lower_limit = 7,
                    nsigma = 2,
                    minimum_points = 10,
                    crowd_radius_arc = None,
                    minimum_frequency_spacing = 0,
                    catalogs = all_catalogs.catalogs, # all
                    #catalogs = [racs_low, racs_mid, racs_high, vlssr, tgss, gleam_300, nvss, wenss, lofar_dr3], # no galactic specific surveys
                    #catalogs = [racs_low, nvss, tgss, vlssr],
                    reference_file = None,
                    anchor_catalog = nvss,
                    )

#### Parameters
debug = False
config = full_config
config.setup()

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





#####################################################################
#### System of equations flux-pair solver declination dependance ####
#####################################################################
# bw = 10
# decs = np.arange(-90, 90+bw, bw)

# cor_matrix = np.zeros((len(config.catalogs), len(config.catalogs), len(decs)))
# weight_matrix = np.zeros((len(config.catalogs), len(config.catalogs), len(decs)))

# all_combinations = get_combinations(config.catalogs, size=2)
# output_width = len(str(len(all_combinations)))
# for i, combination in enumerate(all_combinations):
#     local_cats = [config.catalogs[j] for j in combination]
#     output = compute_flux_correction_factor(local_cats, config, debug=False, anchor_override=0)
    
#     if output is not None:
#         spx, curv, snr, cor, flux, max_sep, p_weight, n_crowd, ra, dec = output
#         print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", f"Matches: {len(spx)}")
        
#         for d, declination in enumerate(decs):
#             dec_bin = (dec >= declination) & (dec < declination + bw)
            
#             tot_wf = calculate_correction_factor_weight(spx[dec_bin], snr[dec_bin], max_sep[dec_bin], p_weight[dec_bin], n_crowd[dec_bin], config)
            
#             filter = tot_wf > 0
#             tot_wf = tot_wf[filter]
#             cor_local = cor[dec_bin][filter]
            
#             if len(tot_wf) > 0:
#                 _, _, py = calculate_1d_peak(cor_local, tot_wf, log=True)
                
#                 y, x = combination
#                 cor_matrix[x, y, d] = py
#                 cor_matrix[y, x, d] = 1.0 / py
                
#                 weight_matrix[y, x, d] = np.sum(tot_wf)
#                 weight_matrix[x, y, d] = np.sum(tot_wf)
#     else:
#         print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", "Matches: None")

# scales = []
# for d in range(len(decs)):
#     s_band = solve_flux_scales_band(cor_matrix[:, :, d], weight_matrix[:, :, d], normalize=True)
#     scales.append(s_band)

# fig, ax = plt.subplots(figsize=(10, 6))
# for i, scale in enumerate(np.array(scales).T):
#     ax.plot(decs, scale, label=config.catalogs[i].name)
# plt.legend()
# plt.xlabel("RA [deg]")
# plt.ylabel("Relative correction factor")
# plt.show()


#################################################
#### System of equations flux-triplet solver ####
#################################################

# N = len(config.catalogs)

# log_ratio_sum  = np.zeros((N, N), dtype=float)
# weight_pair    = np.zeros((N, N), dtype=float)
# beta_slope_sum = np.zeros((N, N), dtype=float)
# beta_weight    = np.zeros((N, N), dtype=float)

# all_combinations = get_combinations(config.catalogs, size=3)
# output_width = len(str(len(all_combinations)))

# for ii, combination in enumerate(all_combinations):
#     local_cats = [config.catalogs[j] for j in combination]

#     # --- Match once ---
#     indices, quality = match_catalogs_2D(local_cats, thres_arc=config.thres_arc, return_quality=True, nsigma=config.nsigma, thres_arc_override=config.thres_arc_override, crowd_radius_arc=config.crowd_radius_arc)
    
#     if len(indices[0]) < config.minimum_points:
#         print(f"({ii+1:{output_width}}/{len(all_combinations)})", f"[{', '.join(f'{c.name:9}' for c in local_cats)}]", "Matches: None")
#         continue

#     # --- All 3 anchors now use the identical source set ---
#     b_vals    = {}
#     beta_vals = {}
#     w_vals    = {}

#     for anchor_pos in range(3):
#         output = compute_flux_correction_factor(
#             local_cats, config, debug=False,
#             anchor_override=anchor_pos,
#             precomputed_indices=indices,
#             precomputed_quality=quality
#         )
#         if output is None:
#             continue

#         spx, curv, snr, cor, flux, max_sep, p_weight, n_crowd, ra, dec = output

#         tot_wf = calculate_correction_factor_weight(spx, snr, max_sep, p_weight, n_crowd, config)
#         valid = tot_wf > 0

#         if np.count_nonzero(valid) < config.minimum_points:
#             continue

#         cor_f   = cor[valid]
#         flux_f  = flux[valid]
#         wf      = tot_wf[valid]

#         log_flux = np.log10(flux_f / cor_f)   # log10(observed anchor flux)
#         log_cor  = np.log10(cor_f)             # log10(extrapolated / observed)

#         coeffs = np.polyfit(log_flux, log_cor, deg=0, w=wf)

#         b_vals[anchor_pos]    = coeffs#coeffs[1]   # log10 offset
#         beta_vals[anchor_pos] = 0#coeffs[0]   # flux slope
#         w_vals[anchor_pos]    = np.sum(wf)

#     print(f"({ii+1:{output_width}}/{len(all_combinations)})",f"[{', '.join(f'{c.name:9}' for c in local_cats)}]")

#     # --- Extract pure pairwise constraints via differences ---
#     # b_a - b_b ≈ log10(scale_a / scale_b)  [third catalog cancels]
#     for pa, pb in [(0, 1), (0, 2), (1, 2)]:
#         if pa not in b_vals or pb not in b_vals:
#             continue

#         cat_a = combination[pa]   # global catalog index
#         cat_b = combination[pb]   # global catalog index

#         b_pair    = b_vals[pa]    - b_vals[pb]    # log10(scale_a / scale_b)
#         beta_pair = beta_vals[pa] - beta_vals[pb] # β_a - β_b

#         # Harmonic mean weight: variance of a difference is 1/w_a + 1/w_b
#         w_pair = (w_vals[pa] * w_vals[pb]) / (w_vals[pa] + w_vals[pb])

#         # Always accumulate in upper triangle (lower index first)
#         if cat_a < cat_b:
#             r, c, sign = cat_a, cat_b,  1.0
#         else:
#             r, c, sign = cat_b, cat_a, -1.0

#         log_ratio_sum[r, c]  += w_pair * sign * b_pair
#         weight_pair[r, c]    += w_pair
#         beta_slope_sum[r, c] += w_pair * sign * beta_pair
#         beta_weight[r, c]    += w_pair

# # --- Build final pairwise matrices ---
# cor_matrix         = np.zeros((N, N), dtype=float)
# weight_matrix      = np.zeros((N, N), dtype=float)
# beta_cor_matrix    = np.zeros((N, N), dtype=float)
# beta_weight_matrix = np.zeros((N, N), dtype=float)

# for i in range(N):
#     for j in range(i + 1, N):
#         w = weight_pair[i, j]
#         if w > 0:
#             r_ij = 10 ** (log_ratio_sum[i, j] / w)   # log10 -> linear ratio
#             cor_matrix[i, j]    = r_ij
#             cor_matrix[j, i]    = 1.0 / r_ij
#             weight_matrix[i, j] = w
#             weight_matrix[j, i] = w

#         w_b = beta_weight[i, j]
#         if w_b > 0:
#             b = beta_slope_sum[i, j] / w_b
#             beta_cor_matrix[i, j]    = 10 **  b
#             beta_cor_matrix[j, i]    = 10 ** -b
#             beta_weight_matrix[i, j] = w_b
#             beta_weight_matrix[j, i] = w_b

# scales = solve_flux_scales(cor_matrix, weight_matrix, normalize=True)
# beta_s = np.log10(solve_flux_scales(beta_cor_matrix, beta_weight_matrix, normalize=True))

# print(f"\n{'Catalog':9}  {'scale':>8}  {'beta':>8}  {'interpretation'}")
# print("-------------------------------------------------------")
# for scale, beta, cat in zip(scales, beta_s, config.catalogs):
#     interp = "faint-biased" if beta > 0.02 else ("bright-biased" if beta < -0.02 else "ok")
#     print(f"{cat.name:9}  {scale:8.5f}  {beta:8.4f}  {interp}")
# print("-------------------------------------------------------")

# print("Weight matrix (pairs with >0 sources):")
# for i in range(N):
#     for j in range(i+1, N):
#         if weight_pair[i, j] > 0:
#             print(f"  {config.catalogs[i].name:10} — {config.catalogs[j].name:10}  w={weight_pair[i,j]:.1f}")

# for i in range(N):
#     for j in range(i+1, N):
#         scale_product = scales[i] * scales[j]
#         beta_sum      = beta_s[i] + beta_s[j]
#         if abs(scale_product - 1.0) < 0.001 and abs(beta_sum) < 0.001:
#             print(f"WARNING: {config.catalogs[i].name} + {config.catalogs[j].name} are conjugate / underconstrained!")

# plt.imshow(weight_pair, origin='lower', norm='log')
# plt.colorbar()
# plt.show()


print(f"Calculations done at: {(perf_counter() - start):.2f} s")
