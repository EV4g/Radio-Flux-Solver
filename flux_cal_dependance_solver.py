import numpy as np
import matplotlib.pyplot as plt
from functions import match_catalogs_2D, get_combinations, get_permutations, solve_flux_scales, calculate_contour_statistics
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
                    nsigma = 3,
                    minimum_points = 10,
                    crowd_radius_arc = None,
                    minimum_frequency_spacing = None,
                    #catalogs = [racs_gal, racs, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar_dr3],
                    catalogs = [racs, tgss, nvss],
                    reference_file = None,
                    anchor_catalog = nvss,
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
#         spx, snr, cor, flux, catw, max_sep, p_weight, n_crowd, ra, dec = output
#         print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", f"Matches: {len(spx)}")
        
#         for d, declination in enumerate(decs):
#             dec_bin = (dec >= declination) & (dec < declination + bw)
            
#             tot_wf = calculate_correction_factor_weight(spx[dec_bin], snr[dec_bin], catw[dec_bin], max_sep[dec_bin], p_weight[dec_bin], n_crowd[dec_bin], config)
            
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

N = len(config.catalogs)

# Keep track
log_ratio_sum  = np.zeros((N, N), dtype=float)
weight_pair    = np.zeros((N, N), dtype=float)
beta_slope_sum = np.zeros((N, N), dtype=float)
beta_weight    = np.zeros((N, N), dtype=float)

#all_combinations = get_combinations(config.catalogs, size=3)
all_combinations = get_permutations(config.catalogs, size=3, only_sorted=False)
output_width = len(str(len(all_combinations)))

for ii, combination in enumerate(all_combinations):
    local_cats = [config.catalogs[j] for j in combination]
    
    # Calculate flux correction w.r.t. the first catalog
    output = compute_flux_correction_factor(local_cats, config, debug=False, anchor_override=0)

    if output is None:
        print(f"({ii+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", "Matches: None")
        continue

    spx, snr, cor, flux, catw, max_sep, p_weight, n_crowd, ra, dec = output
    print(f"({ii+1:{output_width}}/{len(all_combinations)})",f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]",  f"Matches: {len(spx)}")

    # Per-source weighting within this triplet
    tot_wf = calculate_correction_factor_weight(spx, snr, catw, max_sep, p_weight, n_crowd, config)

    valid = (tot_wf > 0)
    if np.count_nonzero(valid) < config.minimum_points:
        continue

    cor_local = cor[valid]
    wf_local  = tot_wf[valid]
    spx_local = spx[valid]
    flux_local = flux[valid]
    
    anchor_global = combination[0]
    ref_global    = combination[1]

    if anchor_global < ref_global:
        i, j = anchor_global, ref_global
        sign = 1.0
    else:
        i, j = ref_global, anchor_global
        sign = -1.0

    # flux_local is extrapolated_flux_fit; anchor observed flux = flux_local / cor_local
    log_flux = np.log10(flux_local / cor_local)
    log_cor  = np.log10(cor_local)

    # weighted linear fit: log_cor = b_ij + beta_ij * log_flux
    coeffs   = np.polyfit(log_flux, log_cor, deg=1, w=wf_local)
    beta_ij  = coeffs[0]   # flux-dependent slope
    b_ij     = coeffs[1]   # global offset

    w_ij = np.sum(wf_local) # sum of correction-factor weights

    log_ratio_sum[i, j]  += w_ij * sign * (-b_ij)
    weight_pair[i, j]    += w_ij
    beta_slope_sum[i, j] += w_ij * sign * beta_ij
    beta_weight[i, j]    += w_ij

# Build final pairwise ratio and weight matrices from accumulated sums
cor_matrix    = np.zeros((N, N), dtype=float)
weight_matrix = np.zeros((N, N), dtype=float)
beta_cor_matrix    = np.zeros((N, N), dtype=float)
beta_weight_matrix = np.zeros((N, N), dtype=float)

for i in range(N):
    for j in range(i + 1, N):
        w_ij = weight_pair[i, j]
        if w_ij > 0:
            r_ij = 10 ** (log_ratio_sum[i, j] / w_ij)
            cor_matrix[i, j]    = r_ij
            cor_matrix[j, i]    = 1.0 / r_ij
            weight_matrix[i, j] = w_ij
            weight_matrix[j, i] = w_ij

        w_beta = beta_weight[i, j]
        if w_beta > 0:
            beta_ij = beta_slope_sum[i, j] / w_beta
            beta_cor_matrix[i, j]    = 10 **  beta_ij
            beta_cor_matrix[j, i]    = 10 ** -beta_ij
            beta_weight_matrix[i, j] = w_beta
            beta_weight_matrix[j, i] = w_beta

scales = solve_flux_scales(cor_matrix, weight_matrix, normalize=True)
beta_s = np.log10(solve_flux_scales(beta_cor_matrix, beta_weight_matrix, normalize=True))

print(f"\n{'Catalog':9}  {'scale':>8}  {'beta':>8}  {'interpretation'}")
print("-------------------------------------------------------")
for scale, beta, cat in zip(scales, beta_s, config.catalogs):
    interp = "faint-biased" if beta > 0.02 else ("bright-biased" if beta < -0.02 else "ok")
    print(f"{cat.name:9}  {scale:8.5f}  {beta:8.4f}  {interp}")
print("-------------------------------------------------------")

print("Weight matrix (pairs with >0 sources):")
for i in range(N):
    for j in range(i+1, N):
        if weight_pair[i, j] > 0:
            print(f"  {config.catalogs[i].name:10} — {config.catalogs[j].name:10}  w={weight_pair[i,j]:.1f}")

for i in range(N):
    for j in range(i+1, N):
        scale_product = scales[i] * scales[j]
        beta_sum      = beta_s[i] + beta_s[j]
        if abs(scale_product - 1.0) < 0.001 and abs(beta_sum) < 0.001:
            print(f"WARNING: {config.catalogs[i].name} + {config.catalogs[j].name} are conjugate / underconstrained!")

plt.imshow(weight_pair, origin='lower', norm='log')
plt.colorbar()
plt.show()


print(f"Calculations done at: {perf_counter() - start} s")