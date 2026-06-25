import argparse
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
from functions import plot_statistics, get_combinations, weighted_bin_stats, weighted_bin_stats_2d, predict_flux
from functions import compute_flux_correction_factor, calculate_correction_factor_weight, biweight_location, report_ignored_cats
from time import perf_counter
from catalog_manager import Catalog, Config, Catalog_set, Output
from joblib import Parallel, delayed
from pathlib import Path
from astropy.io import fits
warnings.filterwarnings("ignore", message=".*(non-interactive|tqdm).*")

try:
    from termcolor import colored
except ImportError:
    print("termcolor not found, ignoring color")
    def colored(str, col): return str

#### all currently implemented survey catalogs
all_catalogs = Catalog_set([
    Catalog("catalogs/vlssr/vlssr_clean.fits",                73.8e6,     "vlssr",      scale=1.1733),
    Catalog("catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",   144.6e6,    "lofar_dr3",  scale=1.0564),
    Catalog("catalogs/tgss/tgss_clean.fits",                  150e6,      "tgss",       scale=1.1125),
    Catalog("catalogs/gleam_x_gp/gleam_x_gp_clean.fits",      200e6,      "gleam_xgp",  scale=1.1337),
    Catalog("catalogs/gleam_300/gleam_300_clean.fits",        300e6,      "gleam_300",  scale=1.1337),
    Catalog("catalogs/wenss/wenss_clean.fits",                325e6,      "wenss",      scale=1.0484),
    Catalog("catalogs/vcss/vcss_clean.fits",                  340e6,      "vcss",       scale=0.9815),
    Catalog("catalogs/txs/txs_clean.fits",                    365e6,      "txs",        scale=0.9524),
    Catalog("catalogs/racs/racs_low_gal_clean.fits",          887.5e6,    "racs_gal",   scale=0.8879),  # the galactic portion of the racs-low survey
    Catalog("catalogs/racs/racs_low_clean.fits",              887.5e6,    "racs_low",   scale=0.8879),  # the rest of the racs-low survey
    Catalog("catalogs/apertif/apertif_clean.fits",            1355e6,     "apertif",    scale=0.9765),
    Catalog("catalogs/meerkat/meerkat_clean.fits",            1359.7e6,   "meerkat",    scale=0.8525),
    Catalog("catalogs/racs/racs_mid_clean.fits",              1367.5e6,   "racs_mid",   scale=0.9486),
    Catalog("catalogs/nvss/nvss_clean.fits",                  1400e6,     "nvss",       scale=1),
    Catalog("catalogs/first/first_clean.fits",                1400e6,     "first",      scale=1),
    Catalog("catalogs/racs/racs_high_clean.fits",             1655.5e6,   "racs_high",  scale=0.9901),
    Catalog("catalogs/vlass/vlass_clean.fits",                3000e6,     "vlass",      scale=0.9915),  # vlass
])

#### Preset -> reference catalog name list
_PRESETS = {
    "all":      [cat.name for cat in all_catalogs],
    "default":  ["vlssr", "lofar_dr3", "tgss", "gleam_300", "wenss", "vcss", "txs", "racs_low", "apertif", "racs_mid", "nvss", "first", "racs_high", "vlass"],
}

#### Frequency units
_FREQ_UNIT_SCALE = {"Hz": 1.0, "MHz": 1e6, "GHz": 1e9}
_CUNIT3_SCALE    = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}

class _TeeWriter:
    """Write to multiple streams (e.g. stdout + a log file)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass

def _resolve_freq(image_path, args_freq, args_freq_unit):
    """Find the frequency axis (CTYPE contains 'FREQ') and return its CRVAL in Hz.
    Falls back to --freq + --freq-unit if no frequency axis is present."""
    with fits.open(image_path) as hdul:
        hdr = hdul[0].header
    for i in range(1, hdr.get("NAXIS", 0) + 1):
        ctype = str(hdr.get(f"CTYPE{i}", "")).upper()
        if "FREQ" in ctype:
            crval = float(hdr[f"CRVAL{i}"])
            cunit = str(hdr.get(f"CUNIT{i}", "Hz")).strip()
            return crval * _CUNIT3_SCALE.get(cunit.upper(), 1.0)
    if args_freq is None:
        raise SystemExit(f"Cannot infer frequency from {image_path}: no axis with CTYPE containing 'FREQ'. Use --freq.")
    return args_freq * _FREQ_UNIT_SCALE[args_freq_unit]

def _is_table_catalog(path):
    """True if the FITS file contains a BINTABLE HDU (i.e. a table catalog, not an image)."""
    with fits.open(path) as hdul:
        return any(hdu.header.get("XTENSION") == "BINTABLE" for hdu in hdul)

def _build_parser():
    p = argparse.ArgumentParser(description="Calibrate a radio image or table catalog against reference catalogs.")
    p.add_argument("catalog",                                                help="Path to FITS image or table catalog (the anchor / unknown).")
    p.add_argument("--scale",                     type=float, default=1,     help="Scale values in the anchor catalog by this amount")
    p.add_argument("--catalogs",                  default="default",         help='Preset name (all, default) or comma-separated catalog list.')
    p.add_argument("--anchor-name",               default=None,              help="Registry name for the anchor (default: input filename stem).")
    p.add_argument("--freq",                      type=float, default=None,  help="Central frequency in --freq-unit; for images, inferred from the FITS FREQ axis when present.")
    p.add_argument("--freq-unit",                 choices=list(_FREQ_UNIT_SCALE), default="Hz")
    p.add_argument("--combination-size",          type=int,   default=3,     help="Set matching complexity as well as fitting D.O.F.")
    p.add_argument("--spectral_damping_factor",   type=float, default=5,     help="Dampen unphysical spectral index outliers")
    p.add_argument("--nsigma",                    type=float, default=2,     help="Sigma range to use for error based matching (default: 2)")
    p.add_argument("--snr-lower-limit",           type=float, default=7,     help="Ignore sources below this SNR limit (default: 7)")
    p.add_argument("--minimum-points",            type=int,   default=3,     help="Ignore matched catalogs sets with matches below this limit (default: 3)")
    p.add_argument("--spectral-index-theory",     type=float, default=-0.8,  help="Theoretical value for spectral index for desired source (default: -0.8)")
    p.add_argument("--minimum-frequency-spacing", type=float, default=100e6, help="Ignore catalog matching with a spacing below threshold (Hz)")
    p.add_argument("--minimum-position-error",    type=float, default=None,  help="Set a minimum position error. Sources with lower error at set to this value (Default: None).")
    p.add_argument("--reference-file",            default=None,              help="Provide reference cutout when giving a large catalog to speed up matching")
    p.add_argument("--spatial-filter",            action="store_true",       help="Pre-filter reference catalogs to the anchor's spatial coverage (with 10%% margin).")
    p.add_argument("--no-reload-cache",           action="store_true",       help="Force PyBDSF to re-run on the anchor image.")
    p.add_argument("--save-plots",                action="store_true",       help="Save inspection plots to disk")
    p.add_argument("--debug",                     action="store_true",       help="Store debug plots per set of matches (slow)")
    p.add_argument("--thres-arc",                 type=float, default=None,  help="Override error based matching with simple thresholding (arcsec)")
    p.add_argument("--n-jobs",                    type=int, default=-1,      help="Number of cores to use, defaults to all of them")
    p.add_argument("--logging",                   action="store_true",       help="Write all output to a log file in --output-dir instead of the terminal.")
    p.add_argument("--output-dir",                default=None,              help="Directory to write plots and logs into (default: current working directory).")
    p.add_argument("--seed",                      type=int, default=None,    help="Seed for the spectra-plot random sample (default: random).")
    return p

def main():
    args = _build_parser().parse_args()
    start = perf_counter()

    # error when choosing wrong combination size parameter
    if args.combination_size < 2 or args.combination_size > 4:
        raise SystemExit(f"--combination-size must be >= 2 and <= 4 (got {args.combination_size}).")

    # check for fits file
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise SystemExit(f"Catalog not found: {catalog_path}")

    is_table = _is_table_catalog(catalog_path)
    if is_table:
        if args.freq is None:
            raise SystemExit(f"Table catalog {catalog_path} has no FITS frequency axis. Pass --freq.")
        freq_hz = args.freq * _FREQ_UNIT_SCALE[args.freq_unit]
    else:
        freq_hz = _resolve_freq(catalog_path, args.freq, args.freq_unit)
    anchor_name = args.anchor_name or catalog_path.stem

    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    elif args.save_plots:
        outdir = Path(f"flux_calibrator_output_{anchor_name}")
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path(".")

    if args.logging:
        sys.stdout = open(outdir / f"{anchor_name}_run.log", "w")

    anchor_cat = Catalog(
        path=str(catalog_path),
        freq_hz=freq_hz,
        name=anchor_name,
        scale=args.scale,
        table=is_table,
        reload_cache=not args.no_reload_cache,
    )

    if args.catalogs in _PRESETS:
        ref_names = _PRESETS[args.catalogs]
    else:
        ref_names = [n.strip() for n in args.catalogs.split(",") if n.strip()]

    available = set(Catalog_set.registry)
    unknown = [n for n in ref_names if n not in available]
    if unknown:
        raise SystemExit(f"Unknown catalog(s): {unknown}\nAvailable: {sorted(available)}")
    if not ref_names:
        raise SystemExit("Reference catalog list is empty.")

    ref_cats = [Catalog_set.registry[n] for n in ref_names]
    all_cats = ref_cats + [anchor_cat]

    config = Config(
        spectral_damping_factor=args.spectral_damping_factor,
        snr_lower_limit=args.snr_lower_limit,
        spectral_index_theory=args.spectral_index_theory,
        minimum_points=args.minimum_points,
        nsigma=args.nsigma,
        crowd_radius_arc=None,
        minimum_frequency_spacing=args.minimum_frequency_spacing,
        minimum_position_error=args.minimum_position_error,
        catalogs=all_cats,
        anchor_catalog=anchor_cat,
        reference_file=args.reference_file,
        spatial_filter=args.spatial_filter,
        thres_arc=args.thres_arc if args.thres_arc is not None else 2,
        thres_arc_override=True  if args.thres_arc is not None else False
    )
    
    config.setup()
    output = Output()

    DEBUG_MODE       = args.debug
    INSPECTION_PLOTS = True
    SAVE_PLOTS       = args.save_plots
    COMBINATION_SIZE = args.combination_size

    if DEBUG_MODE:
        # cutdown catalog plot
        for cat in config.catalogs:
            plt.hist(np.log10(cat.flux), alpha=0.6, bins=25, label=cat.name)
        plt.xlabel("log10(flux/Jy)")
        plt.ylabel("count")
        plt.yscale('log')
        plt.legend()
        if SAVE_PLOTS: plt.savefig(outdir / "debug_flux_distribution.png")
        plt.close('all')

        # catalog as function of position
        for cat in config.catalogs:
            if len(cat.ra) > 0: plt.scatter(cat.ra, cat.dec, s=1, label=cat.name)
        plt.gca().set_box_aspect(1)
        plt.xlabel("RA")
        plt.ylabel("Dec")
        plt.legend(loc='lower left')
        if SAVE_PLOTS: plt.savefig(outdir / "debug_catalog_positions.png")
        plt.close('all')


    print(f"Setup done at: {(perf_counter() - start):.2f} s")

    #########################################
    #### catalog combination auto-looper ####
    #########################################
    all_combinations = get_combinations(config.catalogs, size=COMBINATION_SIZE, required_index=config.anchor_catalog_index, minimum_spacing=config.minimum_frequency_spacing)
    output_width = len(str(len(all_combinations)))
    
    # report catalogs excluded by spacing rules
    report_ignored_cats(all_combinations, config)
    
    print(f"Found {len(all_combinations)} valid combinations")
    print("--------------------------------------------------------")

    # error when no valid catalog-catalog combinations are found after freq constraints
    if len(all_combinations) == 0:
        print(colored("Error: no valid catalog combinations found\n", 'red'))
        sys.exit(1)

    # multithread the main flux correction factor loop
    outputs = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(compute_flux_correction_factor)([config.catalogs[j] for j in combo], config) for combo in all_combinations
    )
    
    if all(val is None for val in outputs):
        print(colored(f"Error: no sources where able to be matched in any of the {len(outputs)} combinations\n", 'red'))
        sys.exit(1)

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
                if SAVE_PLOTS: plt.savefig(outdir / f"{config.anchor_catalog.name}_corr_vs_flux.png")
                plt.close('all')

                # compare fitted spectral index with correction factor
                plt.scatter(spx, cor, c=flux, norm='log')
                plt.yscale('log')
                plt.axvline(config.spectral_index_theory, ls='--', c='k')
                plt.axhline(1, ls='--', c='k')
                plt.colorbar(label='Flux (Jy)')
                plt.ylabel("Flux correction factor")
                plt.xlabel(r"Spectral index $\alpha$")
                plt.title("Flux correction as function of spectral index")
                if SAVE_PLOTS: plt.savefig(outdir / f"{config.anchor_catalog.name}_corr_vs_spx.png")
                plt.close('all')

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
                    path=outdir / "correction_factor_vs_spx.png",
                    show=False,
                    xlabel=r"Fitted spectral index $\alpha$",
                    ylabel="Correction factor",
                    title="Correction factor as function of fitted spectral index\nall catalogs")

    plot_statistics(spectral_index, spectral_curvature, total_weighting_factor,
                    save=SAVE_PLOTS,
                    path=outdir / "spectral_curvature_vs_spx.png",
                    show=False,
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
        min_cor, max_cor = 0.25, 4.0
        mask = (correction_factor > min_cor) & (correction_factor < max_cor) #(correction_factor != np.nan)

        #### weight density as function of location
        fig, ax = plt.subplots(figsize=(7.5, 5))
        plt.hist2d(ras, decs, weights=total_weighting_factor/np.max(total_weighting_factor), bins=(75, 50), cmap='Blues')
        plt.colorbar(label='Cummulative weight / max weight')
        plt.ylabel("DEC (deg)")
        plt.xlabel("RA (deg)")
        if SAVE_PLOTS: plt.savefig(outdir / f"{config.anchor_catalog.name}_weight_density_vs_pos.png")
        plt.close('all')


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
        if SAVE_PLOTS: plt.savefig(outdir / f"{config.anchor_catalog.name}_corr_vs_weightfac.png")
        plt.close('all')


        #### correction factor as function of ra and dec separately
        dec_c, dec_mn, dec_std, dec_sem = weighted_bin_stats(decs[mask], correction_factor[mask], total_weighting_factor[mask], n_bins=50)
        ra_c,  ra_mn,  ra_std,  ra_sem  = weighted_bin_stats(ras[mask],  correction_factor[mask], total_weighting_factor[mask], n_bins=50)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        ax1.plot(dec_c, dec_mn, color='steelblue', lw=2, label='Weighted mean')
        ax1.fill_between(dec_c, dec_mn - dec_std, dec_mn + dec_std, alpha=0.10, color='steelblue', label='±1σ (weighted)')
        ax1.fill_between(dec_c, dec_mn - dec_sem, dec_mn + dec_sem, alpha=0.35, color='steelblue', label='±1μ_err')
        ax1.axhline(1, ls='--', color='black', alpha=0.7)
        ax1.set_xlabel('Dec (deg)')
        ax1.set_ylabel('Correction factor')
        ax1.legend()

        ax2.plot(ra_c, ra_mn, color='tomato', lw=2, label='Weighted mean')
        ax2.fill_between(ra_c, ra_mn - ra_std, ra_mn + ra_std, alpha=0.10, color='tomato', label='±1σ (weighted)')
        ax2.fill_between(ra_c, ra_mn - ra_sem, ra_mn + ra_sem, alpha=0.35, color='tomato', label='±1μ_err')
        ax2.axhline(1, ls='--', color='black', alpha=0.7)
        ax2.set_xlabel('RA (deg)')
        ax2.legend()
        fig.suptitle('Weighted correction factor')
        plt.tight_layout()
        if SAVE_PLOTS: plt.savefig(outdir / f"{config.anchor_catalog.name}_corr_vs_weightfac_radec_dual.png")
        plt.close('all')


        #### correction factor as function of [ra, dec] in 2D
        fig, ax = plt.subplots(figsize=(12, 5))

        real_ticks = np.geomspace(min_cor, max_cor, 5, dtype=float)
        log_ticks  = np.log10(real_ticks).tolist()

        n_pts_2d = np.sum(mask)
        if n_pts_2d < 20000:
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
            cbar = fig.colorbar(scatter, ax=ax, label='Correction factor')

        else:
            # hexbin fallback
            log_cf = np.log10(correction_factor[mask])
            max_log_dev = np.max(np.abs(log_cf))

            hb = ax.hexbin(ras[mask], decs[mask], C=log_cf, gridsize=200, cmap='RdYlGn_r', vmin=-max_log_dev, vmax=max_log_dev, alpha=0.8)
            cbar = fig.colorbar(hb, ax=ax, label='Correction factor')

        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in real_ticks])

        ax.set_xlim(ras[mask].min(), ras[mask].max())
        ax.set_ylim(decs[mask].min(), decs[mask].max())
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.set_title('Correction factor map')
        if SAVE_PLOTS: plt.savefig(outdir / f"{config.anchor_catalog.name}_corr_vs_pos_2d.png")
        plt.close('all')


        #### flux as a function of frequency
        MHz = 1e6
        n_plot = min(1000, len(fitted_flux))
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(fitted_flux), n_plot, replace=False)

        min_freq = np.min([cat.freq for cat in config.catalogs])
        max_freq = np.max([cat.freq for cat in config.catalogs])
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), 100) / MHz
        freq0 = config.anchor_catalog.freq / MHz
        freq_pivot = 100e6 / MHz

        fig, ax = plt.subplots(figsize=(9, 6))

        # individual source spectra
        for i in idx:
            si = spectral_index[i]
            ci = spectral_curvature[i]
            f0 = fitted_flux[i]
            if not np.isfinite(si) or f0 <= 0:
                continue

            # back-compute flux at the log-parabola pivot, then predict full spectrum
            ln_scale = np.log(f0) - (si * np.log(freq0 / freq_pivot) + ci * np.log(freq0 / freq_pivot)**2)
            flux_spec = predict_flux(freqs, freq_pivot, np.exp(ln_scale), si, curvature=ci)
            ax.loglog(freqs, flux_spec, lw=0.3, alpha=0.15, color='gray')

        # aggregate power-law (curvature = 0)
        flux_pl = predict_flux(freqs, freq_pivot, np.exp(np.log(np.median(fitted_flux)) - (mspx * np.log(freq0 / freq_pivot))), mspx, curvature=0)
        ax.loglog(freqs, flux_pl, lw=2.5, color='steelblue', label=f'Power law:  α = {mspx:.3f}')

        # aggregate curved fit
        if np.isfinite(mcur) and abs(mcur) > 1e-6:
            ln_scale_mn = np.log(np.median(fitted_flux)) - (mspx * np.log(freq0 / freq_pivot) + mcur * np.log(freq0 / freq_pivot)**2)
            flux_cv = predict_flux(freqs, freq_pivot, np.exp(ln_scale_mn), mspx, curvature=mcur)
            ax.loglog(freqs, flux_cv, lw=2, ls='--', color='tomato', label=f'Curved:  α = {mspx:.3f},  β = {mcur:.3f}')

        # catalog frequencies
        for cat in config.catalogs:
            ax.axvline(cat.freq / MHz, ls=':', alpha=0.25, color='gray')
            ax.text(cat.freq / MHz, ax.get_ylim()[0] * 1.1, cat.name, rotation=45, fontsize=7, alpha=0.5, ha='right')

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Flux (Jy)')
        ax.set_title(f'Reconstructed spectra ({n_plot} sources) {config.anchor_catalog.name}')
        ax.legend(fontsize=9)
        if SAVE_PLOTS: plt.savefig(outdir / f"{config.anchor_catalog.name}_flux_vs_freq.png")
        plt.close('all')




    print(f"Done at: {(perf_counter() - start):.2f} s")

if __name__ == "__main__":
    main()
