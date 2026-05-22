import warnings
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#from tqdm import tqdm
from functions import plot_statistics, get_combinations, weighted_bin_stats, weighted_bin_stats_2d
from functions import compute_flux_correction_factor, calculate_correction_factor_weight, biweight_location
from time import perf_counter
from catalog_manager import Catalog, Config, Catalog_set, Output
from joblib import Parallel, delayed
warnings.filterwarnings("ignore", message=".*(non-interactive|tqdm).*")

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

start = perf_counter()

#### all currently implemented survey catalogs
all_catalogs = Catalog_set([
    Catalog("/catalogs/vlssr/vlssr_clean.fits",                73.8e6,     "vlssr",      scale=1.2008),
    Catalog("/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",   144.6e6,    "lofar_dr3",  scale=1.0466),
    Catalog("/catalogs/tgss/tgss_clean.fits",                  150e6,      "tgss",       scale=1.1158),
    Catalog("/catalogs/gleam_x_gp/gleam_x_gp_clean.fits",      200e6,      "gleam_xgp",  scale=1.0716),
    Catalog("/catalogs/gleam_300/gleam_300_clean.fits",        300e6,      "gleam_300",  scale=1.1578),
    Catalog("/catalogs/wenss/wenss_clean.fits",                325e6,      "wenss",      scale=1.0426),
    Catalog("/catalogs/racs/racs_low_gal_clean.fits",          887.5e6,    "racs_gal",   scale=0.9068),  # the galactic portion of the racs-low survey
    Catalog("/catalogs/racs/racs_low_clean.fits",              887.5e6,    "racs_low",   scale=0.8852),  # the rest of the racs-low survey
    Catalog("/catalogs/apertif/apertif_clean.fits",            1355e6,     "apertif",    scale=0.9624),
    Catalog("/catalogs/meerkat/meerkat_clean.fits",            1359.7e6,   "meerkat",    scale=0.8525),
    Catalog("/catalogs/racs/racs_mid_clean.fits",              1367.5e6,   "racs_mid",   scale=0.9430),
    Catalog("/catalogs/nvss/nvss_clean.fits",                  1400e6,     "nvss",       scale=1),
    Catalog("/catalogs/racs/racs_high_clean.fits",             1655.5e6,   "racs_high",  scale=0.9860),
])

#### catalogs based on pointings or cutouts
pointing_catalogs = Catalog_set([
    Catalog("/data/raw/catalogs/avgtest-MFS-image_pbcor_catalog.fits",  1284e6,     "meerkat_L_cat",  scale=1),       # meerkat L-band
    Catalog("/catalogs/lofar/lofar_sources_pipeline.fits",              144.6e6,    "lofar",          scale=1),       # LOFAR P282+00
    Catalog("/catalogs/other/cygnus_clean.fits",                        336e6,      "cygnus",         scale=1),       # vla cygnus region
    Catalog("/data/raw/catalogs/lofar_cygnus_6arcsec_catalog.fits",     144.6e6,    "cygnus_6",       scale=1),       # lofar cygnus region 6"
    Catalog("/data/raw/catalogs/lofar_cygnus_20arcsec_catalog.fits",    144.6e6,    "cygnus_20",      scale=1),       # lofar cygnus region 20"
    Catalog("/data/raw/catalogs/lofar_cygnus_53arcsec_catalog.fits",    144.6e6,    "cygnus_53",      scale=1),       # lofar cygnus region 53"
    Catalog("/data/raw/catalogs/lofar_cygnus_53arcsec_int_catalog.fits",144.6e6,    "P312+31",        scale=1),       # lofar cygnus region 53" int
])

#### catalgos based on manual .fits files
manual_catalogs = Catalog_set([
    Catalog("/data/raw/avgtest-MFS-image_pbcor.fits",          1284e6,     "meerkat_L-avgm",      scale=1, table=False),
#    Catalog("/data/raw/test-MFS-image_pbcor.fits",             1284e6,     "meerkat-L-band",      scale=1, table=False),
#    Catalog("/data/raw/lofar_cygnus_6arcsec.fits",             144.6e6,    "lofar_cygnus_6",      scale=1, table=False),
#    Catalog("/data/raw/lofar_cygnus_20arcsec.fits",            144.6e6,    "lofar_cygnus_20",     scale=1, table=False),
#    Catalog("/data/raw/lofar_cygnus_53arcsec.fits",            144.6e6,    "lofar_cygnus_53",     scale=1, table=False),
#    Catalog("/data/raw/P312+31_6arcsec.fits",                  144.6e6,    "P312+31",             scale=1, table=False),
#    Catalog("/data/raw/lofar_cygnus_53arcsec_int.fits",        144.6e6,    "lofar_cygnus_53_int", scale=1, table=False),
])

#### available configurations
lofar_dr3_config = Config(spectral_damping_factor = 5,
                          snr_lower_limit = 7,
                          nsigma = 2.5,
                          minimum_points = 3,
                          crowd_radius_arc = None,
                          minimum_frequency_spacing = 0,
                          catalog_names = ["racs_gal", "racs_low", "racs_mid", "racs_high", "meerkat", "vlssr", "tgss", "gleam_300", "gleam_xgp", "nvss", "wenss", "lofar_dr3", "apertif"],
                          reference_file = None,
                          anchor_catalog_name = "lofar_dr3",
                          )

default_config = Config(spectral_damping_factor = 5,
                        snr_lower_limit = 7,
                        nsigma = 3,
                        minimum_points = 3,
                        crowd_radius_arc = None,
                        minimum_frequency_spacing = 0,
                        catalog_names = ["racs_gal", "meerkat", "tgss", "gleam_300", "gleam_xgp", "lofar"],
                        reference_file = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0],
                        anchor_catalog_name = "lofar",
                        )

cygnus_config = Config(spectral_damping_factor = 5,
                       snr_lower_limit = 7,
                       nsigma = 3,
                       minimum_points = 3,
                       crowd_radius_arc = None,
                       minimum_frequency_spacing = 0,
                       catalog_names = ["racs_low", "vlssr", "tgss", "gleam_300", "nvss", "wenss", "lofar_dr3", "cygnus"],
                       reference_file = np.sort(glob.glob(os.getcwd()+"/data/other/*.fits"))[0],
                       anchor_catalog_name = "cygnus",
                       )

test_config = Config(spectral_damping_factor = 5,
                     spectral_index_theory=-0.8,
                     snr_lower_limit = 7,
                     nsigma = 2,
                     minimum_points = 10,
                     crowd_radius_arc = None,
                     minimum_frequency_spacing = 0,#50e6,
                     catalog_names = ["vlssr", "tgss", "gleam_300", "gleam_xgp", "wenss", "racs_low", "nvss", "racs_mid", "racs_high", "apertif", "meerkat_L_cat"],
                     reference_file = os.getcwd()+"/data/raw/avgtest-MFS-image_pbcor.fits",
                     anchor_catalog_name = "meerkat_L_cat",
                     )

#### Parameters
DEBUG_MODE          = False # per matched combination correction-factor plots
INSPECTION_PLOTS    = True  # additional ra-dec plots
SAVE_PLOTS          = False # save plots to disk
COMBINATION_SIZE    = 3     # number of catalogs to combine per cross-match step [2,3,4]
HIGHER_ORDER_SIMPLE         = False # (wip) instead of using higher order cross-matches to fit spectral index, assume theory and average resulting fluxes
HIGHER_ORDER_SIMPLE_SPACING = None  # (wip) set higher-order-simple minimum frequency spacing above which it is SIMPLE=True

#### setup
config = test_config
config.setup()
output = Output()

if DEBUG_MODE:
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

print(f"Setup done at: {(perf_counter() - start):.2f} s")

#########################################
#### catalog combination auto-looper ####
#########################################
all_combinations = get_combinations(config.catalogs, size=COMBINATION_SIZE, required_index=config.anchor_catalog_index, minimum_spacing=config.minimum_frequency_spacing)
output_width = len(str(len(all_combinations)))

print(f"Found {len(all_combinations)} valid combinations")
print("--------------------------------------------------------")

# multithread the main flux correction factor loop
outputs = Parallel(n_jobs=-1, backend='threading')(
    delayed(compute_flux_correction_factor)([config.catalogs[j] for j in combo], config) for combo in all_combinations
)

for i, (combo, out) in enumerate(zip(all_combinations, outputs)):
    local_cats = [config.catalogs[j] for j in combo]

    if out is not None:
        print(f"({i+1:{output_width}}/{len(all_combinations)})",f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]",f"Matches: {len(out[0])}")
        
        output.add(*out)
        spx, curv, snr, cor, flux, max_sep, p_weight, n_crowd, ra, dec = out
        
        if DEBUG_MODE:
            # compare spectral_index_theory assumption versus fitted spectral indices
            plt.scatter(flux, cor, c=spx)
            plt.yscale('log')
            plt.xscale('log')
            plt.colorbar(label = r"Spectral index $\alpha$")
            plt.xlabel(f"{config.anchor_catalog.name} fitted flux (Jy)")
            plt.ylabel("Correction factor")
            plt.title(f"{config.anchor_catalog.name} "+r"flux, $\alpha$=-"+f"{config.spectral_index_theory} vs fitted")
            if SAVE_PLOTS: plt.savefig(f"{config.anchor_catalog.name}_corr_vs_flux.png")
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
            if SAVE_PLOTS: plt.savefig(f"{config.anchor_catalog.name}_corr_vs_spx.png")
            plt.show()
        
    else:
        print(f"({i+1:{output_width}}/{len(all_combinations)})",f"Completed set [{', '.join(f'{cat.name:9}' for cat in local_cats)}]","Matches:", colored("None", "yellow"))
            
print(f"Flux compute done at {(perf_counter() - start):.2f} seconds")

output.concatenate()
total_weighting_factor = calculate_correction_factor_weight(output, config)
weight_mask = total_weighting_factor > 0
output.apply_mask(weight_mask)
total_weighting_factor = total_weighting_factor[weight_mask]

ras, decs, correction_factor, spectral_index, spectral_curvature, fitted_flux, signal_to_noise, max_separation, point_probability, crowding_parameter = output.return_values()

############################################################################
#### plotting correction factor based on all previous catalog matchings ####
############################################################################
mspx, mcor, mcur = biweight_location(spectral_index, np.log10(correction_factor), spectral_curvature, weights=total_weighting_factor)
mcor = 10**mcor

plot_statistics(spectral_index, correction_factor, total_weighting_factor,
                logy=True, 
                save=SAVE_PLOTS,
                xlabel=r"Fitted spectral index $\alpha$",
                ylabel="Correction factor",
                title="Correction factor as function of fitted spectral index\nall catalogs")

plot_statistics(spectral_index, spectral_curvature, total_weighting_factor,
                save=SAVE_PLOTS,
                xlabel=r"Fitted spectral index $\alpha$",
                ylabel="Spectral curvature",
                title="Spectral curvature as function of fitted spectral index\nall catalogs")

print("--------------------------------------------------------")
print(f"Spectral index: {mspx:.3f}, correction factor: {mcor:.3f}, curvature: {mcur:.3f}, total matches: {len(correction_factor)}")
print("--------------------------------------------------------")

##########################
#### inspection plots ####
##########################
if INSPECTION_PLOTS:
    mask = (correction_factor > 0.1) & (correction_factor < 10) #(correction_factor != np.nan)

    #### weight density as function of location
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plt.hist2d(ras, decs, weights=total_weighting_factor/np.max(total_weighting_factor), bins=(75, 50), cmap='Blues')
    plt.colorbar(label='Cummulative weight / max weight')
    plt.ylabel("DEC (deg)")
    plt.xlabel("RA (deg)")
    if SAVE_PLOTS: plt.savefig(f"{config.anchor_catalog.name}_weight_density_vs_pos.png")
    plt.show()
    
    #### correction factor as function of total weighting factor
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(total_weighting_factor, correction_factor, s=1.5, alpha=0.2)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, ls='--', color='black', alpha=0.5, label='1')
    plt.axhline(mcor, ls='--', color='tomato', label='Fit')
    plt.ylabel("Correction factor")
    plt.xlabel("Total weighting factor")
    plt.legend()
    if SAVE_PLOTS: plt.savefig(f"{config.anchor_catalog.name}_corr_vs_weightfac.png")
    plt.show()
    
    #### correction factor as function of ra and dec separately    
    dec_c, dec_mn, dec_std = weighted_bin_stats(decs[mask], correction_factor[mask], total_weighting_factor[mask], n_bins=50)
    ra_c,  ra_mn,  ra_std  = weighted_bin_stats(ras[mask],  correction_factor[mask], total_weighting_factor[mask], n_bins=50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax1.plot(dec_c, dec_mn, color='steelblue', lw=2, label='Weighted mean')
    ax1.fill_between(dec_c, dec_mn - dec_std, dec_mn + dec_std, alpha=0.25, color='steelblue', label='±1σ (weighted)')
    ax1.axhline(1, ls='--', color='black', alpha=0.7)
    ax1.set_xlabel('Dec (deg)')
    ax1.set_ylabel('Correction factor')
    ax1.legend()
    ax2.plot(ra_c, ra_mn, color='tomato', lw=2, label='Weighted mean')
    ax2.fill_between(ra_c, ra_mn - ra_std, ra_mn + ra_std, alpha=0.25, color='tomato', label='±1σ (weighted)')
    ax2.axhline(1, ls='--', color='black', alpha=0.7)
    ax2.set_xlabel('RA (deg)')
    ax2.legend()
    fig.suptitle('Weighted correction factor')
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig(f"{config.anchor_catalog.name}_corr_vs_weightfac_radec_dual.png")
    plt.show()
    
    #### correction factor as function of [ra, dec] in 2D
    fig, ax = plt.subplots(figsize=(12, 5))

    n_pts_2d = np.sum(mask)
    if n_pts_2d < 5000:
        # voronoi plot
        from scipy.spatial import Voronoi
        from matplotlib.patches import Polygon

        points_2d = np.column_stack([ras[mask], decs[mask]])
        values_2d = correction_factor[mask]

        vor = Voronoi(points_2d)

        ra_min, ra_max = ras[mask].min(), ras[mask].max()
        dec_min, dec_max = decs[mask].min(), decs[mask].max()

        log_values = np.log10(values_2d)
        max_log_dev = np.max(np.abs(log_values))

        for point_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            vertices = vor.vertices[region]
            vertices = np.clip(vertices, [ra_min, dec_min], [ra_max, dec_max])

            log_val = log_values[point_idx]
            color_norm = 0.5 + (log_val / (2 * max_log_dev))
            color_norm = np.clip(color_norm, 0, 1)

            poly = Polygon(vertices, facecolor=plt.cm.RdYlGn_r(color_norm), edgecolor='gray', linewidth=0.2, alpha=0.7)
            ax.add_patch(poly)

        scatter = ax.scatter(ras[mask], decs[mask], c=log_values, cmap='RdYlGn_r', vmin=-max_log_dev, vmax=max_log_dev, s=30, edgecolors='black', linewidth=0.5, zorder=5)
        cbar = fig.colorbar(scatter, ax=ax, label='log10(Correction factor)')

    else:
        # hexbin fallback
        log_cf = np.log10(correction_factor[mask])
        max_log_dev = np.max(np.abs(log_cf))

        hb = ax.hexbin(ras[mask], decs[mask], C=log_cf, gridsize=200, cmap='RdYlGn_r', vmin=-max_log_dev, vmax=max_log_dev, alpha=0.8)
        cbar = fig.colorbar(hb, ax=ax, label='log10(Correction factor)')
    
    ax.set_xlim(ras[mask].min(), ras[mask].max())
    ax.set_ylim(decs[mask].min(), decs[mask].max())
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title('Correction factor map')
    if SAVE_PLOTS: plt.savefig(f"{config.anchor_catalog.name}_corr_vs_pos_2d.png")
    plt.show()


print(f"Done at: {(perf_counter() - start):.2f} s")
