import warnings
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from functions import calculate_contour_statistics, get_combinations, weighted_bin_stats, weighted_bin_stats_2d
from functions import compute_flux_correction_factor, calculate_correction_factor_weight, biweight_location
from time import perf_counter
from catalog_manager import Catalog, Config, Catalog_set
from joblib import Parallel, delayed
warnings.filterwarnings("ignore", message=".*non-interactive.*")

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

start = perf_counter()

#### all available catalogs
all_catalogs = Catalog_set([
    Catalog("/catalogs/racs/racs_low_gal_clean.fits",         887.5e6,    "racs_gal",   scale=0.9068),  # the galactic portion of the racs-low survey
    Catalog("/catalogs/racs/racs_low_clean.fits",             887.5e6,    "racs_low",   scale=0.8852),  # the rest of the racs-low survey
    Catalog("/catalogs/racs/racs_mid_clean.fits",             1367.5e6,   "racs_mid",   scale=0.9430),
    Catalog("/catalogs/racs/racs_high_clean.fits",            1655.5e6,   "racs_high",  scale=0.9860),
    Catalog("/catalogs/meerkat/meerkat_clean.fits",           1359.7e6,   "meerkat",    scale=0.8525),
    Catalog("/catalogs/vlssr/vlssr_clean.fits",               73.8e6,     "vlssr",      scale=1.2008),
    Catalog("/catalogs/tgss/tgss_clean.fits",                 150e6,      "tgss",       scale=1.1158),
    Catalog("/catalogs/gleam_300/gleam_300_clean.fits",       300e6,      "gleam_300",  scale=1.1578),
    Catalog("/catalogs/gleam_x_gp/gleam_x_gp_clean.fits",     200e6,      "gleam_xgp",  scale=1.0716),
    Catalog("/catalogs/nvss/nvss_clean.fits",                 1400e6,     "nvss",       scale=1),
    Catalog("/catalogs/wenss/wenss_clean.fits",               325e6,      "wenss",      scale=1.0426),
    Catalog("/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",  144.6e6,    "lofar_dr3",  scale=1),
    Catalog("/catalogs/lofar/lofar_sources_pipeline.fits",    144.6e6,    "lofar",      scale=1.0457),       # LOFAR P282+00
    Catalog("/catalogs/other/cygnus_clean.fits",              336e6,      "cygnus",     scale=1),       # vla cygnus region
    ])

racs_gal, racs_low, racs_mid, racs_high, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar_dr3, lofar, cygnus = all_catalogs.catalogs

#### available configurations
lofar_dr3_config = Config(spectral_damping_factor = 5,
                          snr_lower_limit = 7,
                          nsigma = 2.5,
                          minimum_points = 3,
                          crowd_radius_arc = None,
                          minimum_frequency_spacing = None,
                          catalogs = [racs_low, racs_gal, meerkat, vlssr, tgss, gleam_300, gleam_xgp, nvss, wenss, lofar_dr3],
                          reference_file = None,
                          anchor_catalog = lofar_dr3,
                          )

default_config = Config(spectral_damping_factor = 5,
                        snr_lower_limit = 7,
                        nsigma = 3,
                        minimum_points = 3,
                        crowd_radius_arc = None,
                        minimum_frequency_spacing = None,
                        catalogs = [racs_gal, meerkat, tgss, gleam_300, gleam_xgp, lofar],
                        reference_file = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0],
                        anchor_catalog = lofar,
                        )

cygnus_config = Config(spectral_damping_factor = 5,
                       snr_lower_limit = 7,
                       nsigma = 3,
                       minimum_points = 3,
                       crowd_radius_arc = None,
                       minimum_frequency_spacing = None,
                       catalogs = [racs_low, vlssr, tgss, gleam_300, nvss, wenss, lofar_dr3, cygnus],
                       reference_file = np.sort(glob.glob(os.getcwd()+"/data/other/*.fits"))[0],
                       anchor_catalog = cygnus,
                       )

small_config = Config(spectral_damping_factor = 5,
                      spectral_index_theory=-0.8,
                       snr_lower_limit = 7,
                       nsigma = 2.5,
                       minimum_points = 3,
                       crowd_radius_arc = None,
                       minimum_frequency_spacing = 0,#50e6,
                       catalogs = all_catalogs.catalogs,
                       #catalogs = [racs_low, racs_gal, vlssr, tgss, gleam_300, gleam_xgp, lofar_dr3, wenss, nvss, racs_mid, racs_high],
                       #catalogs = [vlssr, gleam_300, gleam_xgp, tgss, lofar_dr3],
                       #catalogs = [lofar_dr3, racs_low, meerkat, vlssr, tgss],
                       reference_file = None,
                       anchor_catalog = lofar_dr3,
                       )

#### Parameters
debug = False
inspection_plots = True
#config = lofar_dr3_config
#config = default_config
#config = cygnus_config
config = small_config

config.setup()

print(f"Setup done at: {(perf_counter() - start):.2f} s")
print("------------------------------------------------")

if debug:    
    # cutdown catalog plot
    for cat in config.catalogs:
        plt.hist(np.log10(cat.flux), alpha=0.6, bins=25, label=cat.name)
    plt.xlabel("log10(flux/Jy)")
    plt.ylabel("count")
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    # catalog as function of position
    for cat in config.catalogs:
        if len(cat.ra) > 0: plt.scatter(cat.ra, cat.dec, s=1, label=cat.name)
    plt.gca().set_box_aspect(1)
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend(loc='lower left')
    plt.show()

#### variables
ras                      = [] # positional coordinates
decs                     = [] # positional coordinates
correction_factor_global = [] # ratio between read-out anchor_catalog flux and computed flux
spectral_index_global    = [] # per-source spectral index
spectral_curvature       = [] # per-source spectral curvature
fitted_flux              = [] # anchor_catalog flux based on spectral index extrapolation
signal_to_noise          = [] # signal-to-noise (flux_jy / e_flux_jy)
max_separation           = [] # maximum per-source separation between all three matched catalog positions
point_probability        = [] # probability of points matching
crowding_parameter       = [] # maximum number of neighbours per source within crowd_radius_arc

###################################################
#### catalog three-way combination auto-looper ####
###################################################
all_combinations = get_combinations(config.catalogs, size=3, required_index=config.anchor_catalog_index, minimum_spacing=config.minimum_frequency_spacing)
output_width = len(str(len(all_combinations)))

outputs = Parallel(n_jobs=-1)(delayed(compute_flux_correction_factor)([config.catalogs[j] for j in combo], config) for combo in all_combinations)

for i, (combo, output) in enumerate(zip(all_combinations, outputs)):
    local_cats = [config.catalogs[j] for j in combo]

    if output is not None:
        spx, curv, snr, cor, flux, max_sep, p_weight, n_crowd, ra, dec = output
        print(f"({i+1:{output_width}}/{len(all_combinations)})",f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]",f"Matches: {len(spx)}")

        ras                      += [ra]
        decs                     += [dec]
        correction_factor_global += [cor]
        spectral_index_global    += [spx]
        spectral_curvature       += [curv]
        fitted_flux              += [flux]
        signal_to_noise          += [snr]
        max_separation           += [max_sep]
        point_probability        += [p_weight]
        crowding_parameter       += [n_crowd]
        
        if debug:
            # compare -0.7 assumption versus fitted spectral indices
            plt.scatter(flux, cor, c=spx)
            plt.yscale('log')
            plt.xscale('log')
            plt.colorbar(label = r"Spectral index $\alpha$")
            plt.xlabel(f"{config.anchor_catalog.name} fitted flux (Jy)")
            plt.ylabel("Correction factor")
            plt.title(f"{config.anchor_catalog.name} "+r"flux, $\alpha$=-0.7 vs fitted")
            plt.show()
            
            # compare fitted spectral index with correction factor
            plt.scatter(spx, cor, c=flux, norm='log')
            plt.yscale('log')
            plt.axvline(-0.7, ls='--', c='k')
            plt.axhline(1, ls='--', c='k')
            plt.colorbar(label='Flux (Jy)')
            plt.ylabel("Flux correction factor")
            plt.xlabel(r"Spectral index $\alpha$")
            plt.title("Flux correction as function of spectral index")
            plt.show()
        
    else:
        print(f"({i+1:{output_width}}/{len(all_combinations)})",f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]","Matches:", colored("None", "yellow"))
            
print(f"Flux compute done at {(perf_counter() - start):.2f} seconds")




ras = np.concatenate(ras)
decs = np.concatenate(decs)
correction_factor_global = np.concatenate(correction_factor_global)
spectral_index_global = np.concatenate(spectral_index_global)
spectral_curvature = np.concatenate(spectral_curvature)
fitted_flux = np.concatenate(fitted_flux)
signal_to_noise = np.concatenate(signal_to_noise)
max_separation = np.concatenate(max_separation)
crowding_parameter = np.concatenate(crowding_parameter)
point_probability = np.concatenate(point_probability)

total_weighting_factor = calculate_correction_factor_weight(spectral_index_global,
                                                            signal_to_noise,
                                                            max_separation,
                                                            point_probability,
                                                            crowding_parameter,
                                                            config)

mask = total_weighting_factor > 0

ras                      = ras[mask]
decs                     = decs[mask]
correction_factor_global = correction_factor_global[mask]
spectral_index_global    = spectral_index_global[mask]
spectral_curvature       = spectral_curvature[mask]
fitted_flux              = fitted_flux[mask]
signal_to_noise          = signal_to_noise[mask]
max_separation           = max_separation[mask]
point_probability        = point_probability[mask]
crowding_parameter       = crowding_parameter[mask]

total_weighting_factor   = total_weighting_factor[mask]

############################################################################
#### plotting correction factor based on all previous catalog matchings ####
############################################################################


    
mspx, mcor, mcur = biweight_location(spectral_index_global, correction_factor_global, spectral_curvature, weights=total_weighting_factor)


Xi, Yi, Zi, px, py = calculate_contour_statistics(spectral_index_global, correction_factor_global, total_weighting_factor, logy=True, n=1000)

o = np.argsort(total_weighting_factor)

fig, ax = plt.subplots()
sc = ax.scatter(spectral_index_global[o], correction_factor_global[o], c=total_weighting_factor[o])
ax.contour(Xi, Yi, Zi, levels=6, colors='red', alpha=0.7, linewidths=0.8)
plt.colorbar(sc, label="Combined weighting factor")
plt.yscale('log')

plt.axhline(py, ls="--", color="gray")
plt.axvline(px, ls="--", color="gray")

plt.xlim(np.percentile(spectral_index_global, 1), np.percentile(spectral_index_global, 99))
plt.ylim(np.percentile(correction_factor_global, 1), np.percentile(correction_factor_global, 99))
plt.ylabel("Correction factor")
plt.xlabel(r"Fitted spectral index $\alpha$")
plt.title("Correction factor as function of fitted spectral index\nall catalogs")
plt.show()

print("------------------------------------------------")
print(f"Spectral index: {px:.3f}, correction factor: {py:.3f}, total matches: {len(correction_factor_global)}")
print(f"Spectral index: {mspx:.3f}, correction factor: {mcor:.3f}, total matches: {len(correction_factor_global)}")
print("------------------------------------------------")

##########################
#### inspection plots ####
##########################
if inspection_plots:
    #### position dependant correction factor
    #o = np.argsort(correction_factor_global)
    f = (correction_factor_global[o] > 1e-2) & (correction_factor_global[o] < 1e2)
    plt.scatter(ras[o][f], decs[o][f], c=correction_factor_global[o][f], s=2, norm='log')
    plt.colorbar(label='Correction factor')
    plt.ylabel("DEC (deg)")
    plt.xlabel("RA (deg)")
    plt.show()
    
    #### correction factor as function of total weighting factor
    plt.scatter(total_weighting_factor, correction_factor_global, s=1.5, alpha=0.2)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, ls='--', color='black', alpha=0.5, label='1')
    plt.axhline(py, ls='--', color='tomato', label='Fit')
    plt.ylabel("Correction factor")
    plt.xlabel("Total weighting factor")
    plt.legend()
    plt.show()
    
    #### correction factor as function of ra and dec separately
    mask = (correction_factor_global < 10) & (correction_factor_global > 0.1)
    
    cmean = np.average(correction_factor_global[mask], weights=total_weighting_factor[mask])
    cstd = np.std(correction_factor_global[mask])
    cmin, cmax = max(0.1, cmean - 3 * cstd), cmean + 3 * cstd
    mask &= (correction_factor_global > cmin) & (correction_factor_global < cmax) & (total_weighting_factor > 0)
    
    dec_c, dec_mn, dec_std = weighted_bin_stats(decs[mask], correction_factor_global[mask], total_weighting_factor[mask], n_bins=50)
    ra_c,  ra_mn,  ra_std  = weighted_bin_stats(ras[mask],  correction_factor_global[mask], total_weighting_factor[mask], n_bins=50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(dec_c, dec_mn, color='steelblue', lw=2, label='Weighted mean')
    ax1.fill_between(dec_c, dec_mn - dec_std, dec_mn + dec_std, alpha=0.25, color='steelblue', label='±1σ (weighted)')
    ax1.set_xlabel('Dec (deg)')
    ax1.set_ylabel('Correction factor')
    ax1.legend()
    ax2.plot(ra_c, ra_mn, color='tomato', lw=2, label='Weighted mean')
    ax2.fill_between(ra_c, ra_mn - ra_std, ra_mn + ra_std, alpha=0.25, color='tomato', label='±1σ (weighted)')
    ax2.set_xlabel('RA (deg)')
    ax2.legend()
    fig.suptitle('Weighted correction factor')
    plt.tight_layout()
    plt.show()
    
    #### correction factor as function of [ra, dec] in 2D
    ra_c2, dec_c2, wmean_2d, wstd_2d = weighted_bin_stats_2d(
        ras[mask], decs[mask],
        correction_factor_global[mask],
        total_weighting_factor[mask],
        n_bins=400, m_bins=100
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = ax1.pcolormesh(ra_c2, dec_c2, wmean_2d.T, cmap='RdYlGn_r', shading='auto')
    fig.colorbar(im1, ax=ax1, label='Correction factor')
    ax1.set_xlabel('RA (deg)')
    ax1.set_ylabel('Dec (deg)')
    ax1.set_title('Weighted mean')
    
    im2 = ax2.pcolormesh(ra_c2, dec_c2, wstd_2d.T, cmap='plasma', shading='auto')
    fig.colorbar(im2, ax=ax2, label='Std')
    ax2.set_xlabel('RA (deg)')
    ax2.set_ylabel('Dec (deg)')
    ax2.set_title('Weighted ±1σ')
    
    fig.suptitle('Correction factor map')
    plt.tight_layout()
    plt.show()

    #### plot point densities per catalog
    # for cat in config.catalogs:
    #     plt.hist2d(-cat.ra, cat.dec, bins=(400, 100))
    #     plt.title(cat.name)
    #     plt.show()

print(f"Done at: {(perf_counter() - start):.2f} s")

#### variables
# ras                      = [] # positional coordinates
# decs                     = [] # positional coordinates
# correction_factor_global = [] # ratio between read-out anchor_catalog flux and computed flux
# spectral_index_global    = [] # per-source spectral index
# spectral_curvature       = [] # per-source spectral curvature
# fitted_flux              = [] # anchor_catalog flux based on spectral index extrapolation
# signal_to_noise          = [] # signal-to-noise (flux_jy / e_flux_jy)
# max_separation           = [] # maximum per-source separation between all three matched catalog positions
# point_probability        = [] # probability of points matching
# crowding_parameter       = [] # maximum number of neighbours per source within crowd_radius_arc

###################################################
#### catalog four-way combination auto-looper ####
###################################################
# all_combinations = get_combinations(config.catalogs, size=4, required_index=config.anchor_catalog_index)
# output_width = len(str(len(all_combinations)))
# for i, combination in enumerate(all_combinations):
#     local_cats = [config.catalogs[j] for j in combination]
#     output = compute_flux_correction_factor(local_cats, config, debug=debug)
    
#     if output is not None:
#         spx, curv, snr, cor, flux, max_sep, p_weight, n_crowd, ra, dec = output
#         print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", f"Matches: {len(spx)}")

#         ras                      += [ra]
#         decs                     += [dec]
#         correction_factor_global += [cor]
#         spectral_index_global    += [spx]
#         spectral_curvature       += [curv]
#         fitted_flux              += [flux]
#         signal_to_noise          += [snr]
#         max_separation           += [max_sep]
#         point_probability        += [p_weight]
#         crowding_parameter       += [n_crowd]
#     else:
#         print(f"({i+1:{output_width}}/{len(all_combinations)})", f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]", "Matches: None")

# print(f"Calculations done at: {perf_counter() - start} s")
