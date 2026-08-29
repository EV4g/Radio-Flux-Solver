import os
from astropy.table import Table, Column
import bdsf
import glob
import numpy as np

def simplify(text):
    return "".join(char for char in text if not char.isdigit() and char != "_").lower()

def standardize_catalog_columns(cat, override_units=False, fallback=True):
    # we want to be able to take any inputs, and output with all the same column names
    # and units, for convenience.
    # column names: ra,  dec, e_ra, e_dec, flux_jy, e_flux_jy
    # column units: deg, deg, deg,  deg,   jy,      jy

    # coordinates
    possible_ras    = ["ra", "raj", "raicrs"]
    possible_decs   = ["dec", "decj", "decicrs", "dej", "de", "deicrs"]
    possible_fluxes = ["flux", "fluxjy", "fint", "ftot", "ftotc", "s", "sint", "totalflux", "totalfluxsource", "intflux"]
    #possible_e_flux = ["rms"]
    renamed_ra, renamed_dec, renamed_flux_jy = False, False, False

    ############
    #### ra ####
    ############
    if not 'ra' in cat.colnames:
        # first try exact cases
        for col in cat.colnames:
            if simplify(col) in possible_ras:
                cat.rename_column(col, "ra")
                renamed_ra = True
                break
        
        if not renamed_ra and fallback:
            # fall back to anything with ra and has no e
            for col in cat.colnames:
                if simplify(col).startswith("ra") and "e" not in simplify(col):
                    cat.rename_column(col, "ra")
                    break

    if not 'ra' in cat.colnames:
        raise KeyError(f"No RA column found. Available columns: {cat.colnames}")

    # not returned yet, so must have gone well, check units
    if str(cat['ra'].unit) != 'deg':
        # might be capitalized
        if str(cat['ra'].unit).lower() == 'deg': 
            cat['ra'].unit = 'deg'
        
        # no units; check coverage
        if fallback and (max(cat['ra']) >= 4 and max(cat['ra']) <= 360 and min(cat['ra']) >= 0 and min(cat['ra']) <= 360): 
            cat['ra'].unit = 'deg'
        
        if str(cat['ra'].unit) != 'deg' and not override_units:
            raise KeyError(f"Ambigious values without units in RA column. (min,max): ({min(cat['ra'])}, {max(cat['ra'])})")
        else:
            cat['ra'].unit = 'deg'

    ##############
    #### e_ra ####
    ##############
    if not 'e_ra' in cat.colnames:
        # anything with ra and starts with e
        for col in cat.colnames:
            if simplify(col).startswith("e") and "ra" in simplify(col):
                cat.rename_column(col, "e_ra")
                break

    if not 'e_ra' in cat.colnames:
        raise KeyError(f"No e_RA column found. Available columns: {cat.colnames}")

    # not returned yet, so must have gone well, check units
    if str(cat['e_ra'].unit) != 'deg':
        # might be capitalized
        if str(cat['e_ra'].unit).lower() == 'deg': 
            cat['e_ra'].unit = 'deg'

        # might be seconds instead of deg or arcsec
        if str(cat['e_ra'].unit) == 's':
            cat['e_ra'] *= 15 / 3600   # 1s RA = 15 arcsec
            cat['e_ra'].unit = 'deg'
        
        # might be arcsec; also if values are huge
        if fallback and (str(cat['e_ra'].unit) == 'arcsec' or min(cat['e_ra'][cat['e_ra']>0]) >= 2):
            cat['e_ra'] /= 3600
            cat['e_ra'].unit = 'deg'

        # no units yet; check if between 1" and 10" 
        if fallback and (str(cat['e_ra'].unit) != 'deg' and np.mean(cat['e_ra']) * 3600 > 1 and np.mean(cat['e_ra']) * 3600 < 10):
            cat['e_ra'].unit = 'deg'
        
        if str(cat['e_ra'].unit) != 'deg' and not override_units:
            raise KeyError(f"Ambigious values without units in e_RA column. (min,max): ({min(cat['e_ra'])}, {max(cat['e_ra'])})")
        else:
            cat['e_ra'].unit = 'deg'

    #############
    #### dec ####
    #############
    if not 'dec' in cat.colnames:
        # first try exact cases
        for col in cat.colnames:
            if simplify(col) in possible_decs:
                cat.rename_column(col, "dec")
                renamed_dec = True
                break
        
        if not renamed_dec and fallback:
            # fall back to anything with dec
            for col in cat.colnames:
                if simplify(col).startswith("dec"):
                    cat.rename_column(col, "dec")
                    break

    if not 'dec' in cat.colnames:
        raise KeyError(f"No DEC column found. Available columns: {cat.colnames}")

    # not returned yet, so must have gone well, check units
    if str(cat['dec'].unit) != 'deg':
        # might be capitalized
        if str(cat['dec'].unit).lower() == 'deg': 
            cat['dec'].unit = 'deg'
        
        # no units; check coverage
        if fallback and (max(cat['dec']) >= -90 and max(cat['dec']) <= 90 and min(cat['dec']) >= -90 and min(cat['dec']) <= 90): 
            cat['dec'].unit = 'deg'

        if str(cat['dec'].unit) != 'deg' and not override_units:
            raise KeyError(f"Ambigious values without units in DEC column. (min,max): ({min(cat['dec'])}, {max(cat['dec'])})")
        else:
            cat['dec'].unit = 'deg'

    ###############
    #### e_dec ####
    ###############
    if not 'e_dec' in cat.colnames:
        # fall back to anything with dec and starts with e
        for col in cat.colnames:
            if simplify(col).startswith("e") and "de" in simplify(col):
                cat.rename_column(col, "e_dec")
                break

    if not 'e_dec' in cat.colnames:
        raise KeyError(f"No e_DEC column found. Available columns: {cat.colnames}")

    # not returned yet, so must have gone well, check units
    if str(cat['e_dec'].unit) != 'deg':
        # might be capitalized
        if str(cat['e_dec'].unit).lower() == 'deg': 
            cat['e_dec'].unit = 'deg'

        # might be seconds instead of deg or arcsec
        if str(cat['e_dec'].unit) == 's':
            cat['e_dec'] *= 15 / 3600   # 1s RA = 15 arcsec
            cat['e_dec'].unit = 'deg'

        # might be arcsec; also if values are huge
        if fallback and str(cat['e_dec'].unit) == 'arcsec' or min(cat['e_dec'][cat['e_dec']>0]) >= 2:
            cat['e_dec'] /= 3600
            cat['e_dec'].unit = 'deg'
                
        # no units yet; check if between 1" and 10" 
        if fallback and (str(cat['e_dec'].unit) != 'deg' and np.mean(cat['e_dec']) * 3600 > 1 and np.mean(cat['e_dec']) * 3600 < 10):
            cat['e_dec'].unit = 'deg'
        
        if str(cat['e_dec'].unit) != 'deg' and  not override_units:
            raise KeyError(f"Ambigious values without units in e_DEC column. (min,max): ({min(cat['e_dec'])}, {max(cat['e_dec'])})")
        else:
            cat['e_dec'].unit = 'deg'

    #################
    #### flux_jy ####
    #################
    if not 'flux_jy' in cat.colnames:
        # first try exact cases
        for col in cat.colnames:
            if simplify(col) in possible_fluxes:
                cat.rename_column(col, "flux_jy")
                renamed_flux_jy = True
                break
    
        if not renamed_flux_jy and fallback:
            # fall back to anything with flux-ish
            for col in cat.colnames:
                if (simplify(col).startswith("s") or simplify(col).startswith('int')) and "e" not in simplify(col):
                    cat.rename_column(col, "flux_jy")
                    break

    if not 'flux_jy' in cat.colnames:
        raise KeyError(f"No flux column found. Available columns: {cat.colnames}")

    # not returned yet, so must have gone well, check units
    if str(cat['flux_jy'].unit) != 'Jy':
        # might be capitalized differently
        if str(cat['flux_jy'].unit).lower() == 'jy':
            cat['flux_jy'].unit = 'Jy'

        # might be mJy
        if str(cat['flux_jy'].unit).lower() == 'mjy':
            cat['flux_jy'] /= 1000
            cat['flux_jy'].unit = 'Jy'
        
        if str(cat['flux_jy'].unit) != 'Jy' and not override_units:
            raise KeyError(f"Ambigious values without units in flux column. (min,max): ({min(cat['flux_jy'])}, {max(cat['flux_jy'])})")
        else:
            cat['flux_jy'].unit = 'Jy'

    ###################
    #### e_flux_jy ####
    ###################
    if not 'e_flux_jy' in cat.colnames:
        # fall back to anything with flux-ish
        for col in cat.colnames:
            if simplify(col).startswith("e") and ("s" in simplify(col) or "f" in simplify(col)):
                cat.rename_column(col, "e_flux_jy")
                break

    if not 'e_flux_jy' in cat.colnames:
        raise KeyError(f"No e_flux column found. Available columns: {cat.colnames}")

    # not returned yet, so must have gone well, check units
    if str(cat['e_flux_jy'].unit) != 'Jy':
        # might be capitalized
        if str(cat['e_flux_jy'].unit).lower() == 'Jy':
            cat['e_flux_jy'].unit = 'Jy'

        # might be mJy
        if str(cat['e_flux_jy'].unit).lower() == 'mjy':
            cat['e_flux_jy'] /= 1000
            cat['e_flux_jy'].unit = 'Jy'
        
        if str(cat['e_flux_jy'].unit) != 'Jy' and not override_units:
            raise KeyError(f"Ambigious values without units in flux column. (min,max): ({min(cat['e_flux_jy'])}, {max(cat['e_flux_jy'])})")
        else:
            cat['e_flux_jy'].unit = 'Jy'


    # assert all operations went according to plan
    assert str(cat['ra'].unit) == 'deg'     and str(cat['dec'].unit) == 'deg'
    assert str(cat['e_ra'].unit) == 'deg'   and str(cat['e_dec'].unit) == 'deg'
    assert str(cat['flux_jy'].unit) == 'Jy' and str(cat['e_flux_jy'].unit) == 'Jy'

    # disregard any other columns to save memory
    cat.keep_columns(['ra', 'dec', 'e_ra', 'e_dec', 'flux_jy', 'e_flux_jy'])
    
    return cat


if __name__ == "__main__":
    # get calagogs
    # racs_full = Table.read(os.getcwd()+"/catalogs/racs/racs.fits")
    # racs_gal  = Table.read(os.getcwd()+"/catalogs/racs/RACS_DR1_Sources_GalacticRegion_v2021_08.xml")
    # meerkat   = Table.read(os.getcwd()+"/catalogs/meerkat/smgps_moment0_5beam_5sigma_510599row_compact_source_catalogue.fits")
    # vlssr     = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_full.csv")
    # tgss      = Table.read(os.getcwd()+"/catalogs/tgss/TGSSADR1_7sigma_catalog.fits")
    # gleam     = Table.read(os.getcwd()+"/catalogs/gleam_300/GLEAM300_source_catalogue.fits")
    # gleam_xgp = Table.read(os.getcwd()+"/catalogs/gleam_x_gp/gleam_x_gp.fit")
    # nvss      = Table.read(os.getcwd()+"/catalogs/nvss/NVSS.fits")
    # wenss     = Table.read(os.getcwd()+"/catalogs/wenss/WENSS.fits")
    # lofar_dr3 = Table.read(os.getcwd()+"/catalogs/lofar/LoTSS_DR3_v1.0.srl.fits")
    # lofar     = Table.read(os.getcwd()+'/catalogs/lofar/lofar_sources_pipeline.fits')
    # cygnus    = Table.read(os.getcwd()+'/catalogs/other/cygnus_sources.fits')
    # racs_mid  = Table.read(os.getcwd()+"/catalogs/racs/RACS-mid1_sources.xml")
    # racs_high = Table.read(os.getcwd()+"/catalogs/racs/RACS-high_sources.xml")
    # apertif   = Table.read(os.getcwd()+"/catalogs/apertif/apertif.fits")
    # vlass     = Table.read(os.getcwd()+"/catalogs/vlass/vlass.fit")
    # txs       = Table.read(os.getcwd()+"/catalogs/txs/TXS.fit")
    # vcss      = Table.read(os.getcwd()+"/catalogs/vcss/VCSS.fit")
    # first     = Table.read(os.getcwd()+"/catalogs/first/first.fits")
    
    
    #### first
    # first.rename_column("RAJ2000", "ra")
    # first.rename_column("DEJ2000", "dec")
    # first.rename_column("Fint", "flux_jy")
    # first.rename_column("Rms", "e_flux_jy")
    
    # # error = size * (1/snr + 1/20) * arcsec
    # size = (first["Maj"] + first["Min"]) * 0.5 # roughly
    # snr = (first["flux_jy"] - 0.25) / first["e_flux_jy"] # both still in mJy
    # error = size * (1/snr + 1/20) / 1.645 # size is already in arcsec
    
    # first["e_ra"] = error / 3600
    # first["e_dec"] = error / 3600
    # first["e_ra"].unit = 'deg'
    # first["e_dec"].unit = 'deg'
    
    # # currently no units
    # first["ra"].unit = 'deg'
    # first["dec"].unit = 'deg'
    
    # if str(first['flux_jy'].unit) == 'mJy':
    #     first['flux_jy'] *= 1e-3
    #     first['flux_jy'].unit = 'Jy'
    #     first['e_flux_jy'] *= 1e-3
    #     first['e_flux_jy'].unit = 'Jy'
    
    # assert str(first['ra'].unit) == 'deg' and str(first['dec'].unit) == 'deg'
    # assert str(first['e_ra'].unit) == 'deg' and str(first['e_dec'].unit) == 'deg'
    # assert str(first['flux_jy'].unit) == 'Jy' and str(first['e_flux_jy'].unit) == 'Jy'
    
    # first.write(os.getcwd()+"/catalogs/first/first_clean.fits", overwrite=True)
    
    
    #### vcss
    # vcss.rename_column("RAJ2000", "ra")
    # vcss.rename_column("DEJ2000", "dec")
    # vcss.rename_column("e_RAJ2000", "e_ra")
    # vcss.rename_column("e_DEJ2000", "e_dec")
    # vcss.rename_column("Ftotc", "flux_jy")
    # vcss.rename_column("e_Ftot", "e_flux_jy")
    
    # if str(vcss['flux_jy'].unit) == 'mJy':
    #     vcss['flux_jy'] *= 1e-3
    #     vcss['flux_jy'].unit = 'Jy'
    #     vcss['e_flux_jy'] *= 1e-3
    #     vcss['e_flux_jy'].unit = 'Jy'
    
    # assert str(vcss['ra'].unit) == 'deg' and str(vcss['dec'].unit) == 'deg'
    # assert str(vcss['e_ra'].unit) == 'deg' and str(vcss['e_dec'].unit) == 'deg'
    # assert str(vcss['flux_jy'].unit) == 'Jy' and str(vcss['e_flux_jy'].unit) == 'Jy'
    
    # vcss.write(os.getcwd()+"/catalogs/vcss/vcss_clean.fits", overwrite=True)
    
    #### txs
    # txs.rename_column("_RAJ2000", "ra")
    # txs.rename_column("_DEJ2000", "dec")
    # txs.rename_column("e_RA1950", "e_ra")
    # txs.rename_column("e_DE1950", "e_dec")
    # txs.rename_column("S365", "flux_jy")
    # txs.rename_column("e_S365", "e_flux_jy")
    
    # if str(txs['e_ra'].unit) == 's':
    #     txs['e_ra'] *= 15 / 3600
    #     txs['e_ra'].unit = 'deg'
    # if str(txs['e_dec'].unit) == 'arcsec':
    #     txs['e_dec'] /= 3600
    #     txs['e_dec'].unit = 'deg'
    
    # txs.keep_columns(['ra', 'dec', 'e_ra', 'e_dec', 'flux_jy', 'e_flux_jy'])
    
    # assert str(txs['ra'].unit) == 'deg' and str(txs['dec'].unit) == 'deg'
    # assert str(txs['e_ra'].unit) == 'deg' and str(txs['e_dec'].unit) == 'deg'
    # assert str(txs['flux_jy'].unit) == 'Jy' and str(txs['e_flux_jy'].unit) == 'Jy'
    
    # txs.write(os.getcwd()+"/catalogs/txs/txs_clean.fits", overwrite=True)
    
    #### vlass
    # vlass.rename_column("RAJ2000", "ra")
    # vlass.rename_column("DEJ2000", "dec")
    # vlass.rename_column("e_RAJ2000", "e_ra")
    # vlass.rename_column("e_DEJ2000", "e_dec")
    # vlass.rename_column("Ftot", "flux_jy")
    # vlass.rename_column("e_Ftot", "e_flux_jy")
    
    # if str(vlass['flux_jy'].unit) == 'mJy':
    #     vlass['flux_jy'] *= 1e-3
    #     vlass['e_flux_jy'] *= 1e-3
    #     vlass['flux_jy'].unit = 'Jy'
    #     vlass['e_flux_jy'].unit = 'Jy'
    
    # if str(vlass['e_ra'].unit) == 'arcsec':
    #     vlass['e_ra'] /= 3600
    #     vlass['e_ra'].unit = 'deg'
    # if str(vlass['e_dec'].unit) == 'arcsec':
    #     vlass['e_dec'] /= 3600
    #     vlass['e_dec'].unit = 'deg'
    
    # assert str(vlass['ra'].unit) == 'deg' and str(vlass['dec'].unit) == 'deg'
    # assert str(vlass['flux_jy'].unit) == 'Jy' and str(vlass['e_flux_jy'].unit) == 'Jy'
    
    # vlass.write("vlass_clean.fits", overwrite=True)
    
    #### apertif
    # apertif.rename_column("RAICRS", "ra")
    # apertif.rename_column("DEICRS", "dec")
    # apertif.rename_column("e_RAICRS", "e_ra")
    # apertif.rename_column("e_DEICRS", "e_dec")
    # apertif.rename_column('Sint', 'flux_jy')
    # apertif.rename_column('e_Sint', 'e_flux_jy')
    
    # if str(apertif['flux_jy'].unit) == 'mJy':
    #     apertif['flux_jy'] *= 1e-3
    #     apertif['e_flux_jy'] *= 1e-3
    #     apertif['flux_jy'].unit = 'Jy'
    #     apertif['e_flux_jy'].unit = 'Jy'
        
    # if str(apertif['e_ra'].unit) == 'arcsec':
    #     apertif['e_ra'] /= 3600
    #     apertif['e_ra'].unit = 'deg'
    # if str(apertif['e_dec'].unit) == 'arcsec':
    #     apertif['e_dec'] /= 3600
    #     apertif['e_dec'].unit = 'deg'
    
    # apertif.write("apertif_clean.fits", overwrite=True)
    
    #### racs_mid
    # racs_mid.rename_column("RA", "ra")
    # racs_mid.rename_column("Dec_corr", "dec")
    # racs_mid.rename_column("E_RA", "e_ra")
    # racs_mid.rename_column("E_Dec", "e_dec")
    # racs_mid.rename_column('Total_flux', 'flux_jy')
    # racs_mid.rename_column('E_Total_flux', 'e_flux_jy')
    
    # if str(racs_mid['flux_jy'].unit) == 'mJy':
    #     racs_mid['flux_jy'] *= 1e-3
    #     racs_mid['e_flux_jy'] *= 1e-3
    #     racs_mid['flux_jy'].unit = 'Jy'
    #     racs_mid['e_flux_jy'].unit = 'Jy'
    
    # if str(racs_mid["e_ra"].unit) == 'arcsec':
    #         racs_mid['e_ra'] /= 3600
    #         racs_mid['e_ra'].unit = 'deg'
    #         racs_mid['e_dec'] /= 3600
    #         racs_mid['e_dec'].unit = 'deg'
    
    # racs_mid_out = Table()
    # for col in racs_mid.colnames:
    #     data = np.array(racs_mid[col])  # strips Quantity/mixin type
    #     unit = str(racs_mid[col].unit) if racs_mid[col].unit else None
    #     racs_mid_out[col] = Column(data, unit=unit)
    
    # racs_mid_out.write("racs_mid_clean.fits", overwrite=True)
    
    #### racs_high ####
    # racs_high.rename_column("RA", "ra")
    # racs_high.rename_column("Dec_corr", "dec")
    # racs_high.rename_column("E_RA", "e_ra")
    # racs_high.rename_column("E_Dec", "e_dec")
    # racs_high.rename_column('Total_flux', 'flux_jy')
    # racs_high.rename_column('E_Total_flux', 'e_flux_jy')
    
    # if str(racs_high['flux_jy'].unit) == 'mJy':
    #     racs_high['flux_jy'] *= 1e-3
    #     racs_high['e_flux_jy'] *= 1e-3
    #     racs_high['flux_jy'].unit = 'Jy'
    #     racs_high['e_flux_jy'].unit = 'Jy'
    
    # if str(racs_high["e_ra"].unit) == 'arcsec':
    #         racs_high['e_ra'] /= 3600
    #         racs_high['e_ra'].unit = 'deg'
    #         racs_high['e_dec'] /= 3600
    #         racs_high['e_dec'].unit = 'deg'
    
    # racs_high_out = Table()
    # for col in racs_high.colnames:
    #     data = np.array(racs_high[col])  # strips Quantity/mixin type
    #     unit = str(racs_high[col].unit) if racs_high[col].unit else None
    #     racs_high_out[col] = Column(data, unit=unit)
    
    # racs_high_out.write("racs_high_clean.fits", overwrite=True)
    
    #### cygnus ####
    # cygnus.rename_column("RA", "ra")
    # cygnus.rename_column("DEC", "dec")
    # cygnus.rename_column("E_RA", "e_ra")
    # cygnus.rename_column("E_DEC", "e_dec")
    # cygnus.rename_column('Total_flux', 'flux_jy')
    # cygnus.rename_column('E_Total_flux', 'e_flux_jy')
    
    # if str(cygnus['flux_jy'].unit) == 'mJy':
    #     cygnus['flux_jy'] *= 1e-3
    #     cygnus['e_flux_jy'] *= 1e-3
    #     cygnus['flux_jy'].unit = 'Jy'
    #     cygnus['e_flux_jy'].unit = 'Jy'
    
    # cygnus.write("cygnus_clean.fits", overwrite=True)
    
    #### lofar-dr3 ####
    # lofar_dr3.rename_column("RA", "ra")
    # lofar_dr3.rename_column("DEC", "dec")
    # lofar_dr3.rename_column("E_RA", "e_ra")
    # lofar_dr3.rename_column("E_DEC", "e_dec")
    # lofar_dr3.rename_column('Total_flux', 'flux_jy')
    # lofar_dr3.rename_column('E_Total_flux', 'e_flux_jy')
    
    # # LOFAR-DR3 uses mJy, pybdsf uses Jy, force everything to Jy
    # if str(lofar_dr3['flux_jy'].unit) == 'mJy':
    #     lofar_dr3['flux_jy'] *= 1e-3
    #     lofar_dr3['e_flux_jy'] *= 1e-3
    #     lofar_dr3['flux_jy'].unit = 'Jy'
    #     lofar_dr3['e_flux_jy'].unit = 'Jy'
    
    # if str(lofar_dr3['e_ra'].unit) == 'arcsec':
    #     lofar_dr3['e_ra'] /= 3600
    #     lofar_dr3['e_ra'].unit = 'deg'
    # if str(lofar_dr3['e_dec'].unit) == 'arcsec':
    #     lofar_dr3['e_dec'] /= 3600
    #     lofar_dr3['e_dec'].unit = 'deg'
    
    # lofar_dr3.write('LoTSS_DR3_v1.0.srl_clean.fits', overwrite=True)
    
    # from astroquery.vizier import Vizier
    # v = Vizier(row_limit=-1)
    # catalogs = v.get_catalogs("VIII/97")
    # vlssr = catalogs[0]
    # vlssr.write("vlssr_full.csv", format="ascii.csv", overwrite=True)
    
    #### lofar ####
    # lofar_files = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0]
    # img = bdsf.process_image(
    #     lofar_files,
    #     thresh_isl=3.0,       # island threshold (sigma)
    #     thresh_pix=5.0,       # peak detection threshold (sigma)
    #     rms_box=(100, 25),    # (box_size, step_size) for rms map; tune to your image
    #     beam=(get_beam_size(lofar_files)),  # (maj_deg, min_deg, PA)
    # )
    
    # lofar.rename_column("RA", "ra")
    # lofar.rename_column("DEC", "dec")
    # lofar.rename_column("E_RA", "e_ra")
    # lofar.rename_column("E_DEC", "e_dec")
    # lofar.rename_column('Total_flux', 'flux_jy')
    # lofar.rename_column('E_Total_flux', 'e_flux_jy')
    
    # if str(lofar['flux_jy'].unit) == 'mJy':
    #     lofar['flux_jy'] *= 1e-3
    #     lofar['e_flux_jy'] *= 1e-3
    #     lofar['flux_jy'].unit = 'Jy'
    #     lofar['e_flux_jy'].unit = 'Jy'
    
    # img.write_catalog(outfile="lofar_sources_pipeline.fits", format="fits", catalog_type="srl", clobber=True)
    
    #### gleam-x gp
    # gleam_xgp.rename_column("RAJ2000", "ra")
    # gleam_xgp.rename_column("e_RAJ2000", "e_ra")
    # gleam_xgp.rename_column("DEJ2000", "dec")
    # gleam_xgp.rename_column("e_DEJ2000", "e_dec")
    
    # for column in gleam_xgp.colnames:
    #     begin = column[:4]
    #     if begin == "Fint":
    #         new_col_name = "flux_jy"
    #         if len(column) > 4: new_col_name += "_"+column[4:]
    #         gleam_xgp.rename_column(column, new_col_name)
    #     elif begin == "e_Fi":
    #         new_col_name = "e_flux_jy"
    #         if len(column) > 6: new_col_name += "_"+column[6:]
    #         gleam_xgp.rename_column(column, new_col_name)
    
    # gleam_xgp.write("gleam_x_gp_clean.fits")
    
    #### racs_gal ####
    # racs_gal['flux_jy'] = racs_gal['Total_flux_Source'] * 1e-3
    # racs_gal['e_flux_jy'] = racs_gal['E_Total_flux_Source'] * 1e-3
    # racs_gal['flux_jy'].unit = 'Jy'
    # racs_gal['e_flux_jy'].unit = 'Jy'
    # racs_gal.rename_column("RA", "ra")
    # racs_gal.rename_column("Dec", "dec")
    # racs_gal.rename_column("E_RA", "e_ra")
    # racs_gal.rename_column("E_Dec", "e_dec")
    
    # if str(racs_gal['e_ra'].unit) == 'arcsec':
    #     racs_gal['e_ra'] /= 3600
    #     racs_gal['e_ra'].unit = 'deg'
    # if str(racs_gal['e_dec'].unit) == 'arcsec':
    #     racs_gal['e_dec'] /= 3600
    #     racs_gal['e_dec'].unit = 'deg'
    
    # racs_gal_out = Table()
    # for col in racs_gal.colnames:
    #     data = np.array(racs_gal[col])  # strips Quantity/mixin type
    #     unit = str(racs_gal[col].unit) if racs_gal[col].unit else None
    #     racs_gal_out[col] = Column(data, unit=unit)
    
    # racs_gal_out.write("racs_gal_clean.fits", overwrite=True)
    
    #### racs full ####
    # racs_full['flux_jy'] = racs_full['Ftot'] * 1e-3
    # racs_full['e_flux_jy'] = racs_full['s_Ftot'] * 1e-3
    # racs_full['flux_jy'].unit = 'Jy'
    # racs_full['e_flux_jy'].unit = 'Jy'
    # racs_full.rename_column("RAJ2000", "ra")
    # racs_full.rename_column("DEJ2000", "dec")
    # racs_full.rename_column("e_RAJ2000", "e_ra")
    # racs_full.rename_column("e_DEJ2000", "e_dec")
    
    # if str(racs_full['e_ra'].unit) == 'arcsec':
    #     racs_full['e_ra'] /= 3600
    #     racs_full['e_ra'].unit = 'deg'
    # if str(racs_full['e_dec'].unit) == 'arcsec':
    #     racs_full['e_dec'] /= 3600
    #     racs_full['e_dec'].unit = 'deg'
    
    # racs_full.write("racs_full_clean.fits", overwrite=True)
    
    
    
    #### meerkat ####
    # meerkat['flux_jy'] = meerkat['Fint'] * 1e-3
    # meerkat['e_flux_jy'] = meerkat['e_Fint'] * 1e-3
    # meerkat['flux_jy'].unit = 'Jy'
    # meerkat['e_flux_jy'].unit = 'Jy'
    # meerkat.rename_column("e_GLON", "e_ra")
    # meerkat.rename_column("e_GLAT", "e_dec")
    # meerkat.rename_column("RAJ2000", "ra")
    # meerkat.rename_column("DEJ2000", "dec")
    # meerkat.write("meerkat_clean.fits", overwrite=True)
    
    #### vlssr ####
    # DEG_TO_ARCSEC = 3600.0
    # VLSSR_BEAM_ARCSEC = 80.0  # circular restoring beam
    
    # src_maj = vlssr["MajAx"] * DEG_TO_ARCSEC
    # src_min = vlssr["MinAx"] * DEG_TO_ARCSEC
    
    # vlssr['flux_jy'] = vlssr["Sp"] * (src_maj * src_min) / (VLSSR_BEAM_ARCSEC ** 2)
    # vlssr['e_flux_jy'] = vlssr["e_Sp"] * (src_maj * src_min) / (VLSSR_BEAM_ARCSEC ** 2)
    # vlssr['flux_jy'].unit = 'Jy'
    # vlssr['e_flux_jy'].unit = 'Jy'
    
    # vlssr.rename_column("ra_deg", "ra")
    # vlssr.rename_column("dec_deg", "dec")
    # vlssr['e_ra'] = np.ones_like(vlssr['ra']) * 3.5 / DEG_TO_ARCSEC
    # vlssr['e_dec'] = np.ones_like(vlssr['dec']) * 3.5 / DEG_TO_ARCSEC
    
    # vlssr.write("vlssr_clean.fits", overwrite=True)
    
    #### tgss ####
    # tgss.rename_column('RA', 'ra')
    # tgss.rename_column('DEC', 'dec')
    # tgss.rename_column('E_RA', 'e_ra')
    # tgss.rename_column('E_DEC', 'e_dec')
    # tgss.rename_column('Total_flux', 'flux_jy')
    # tgss.rename_column('E_Total_flux', 'e_flux_jy')
    
    # tgss['flux_jy'] *= 1e-3
    # tgss['e_flux_jy'] *= 1e-3
    # tgss['flux_jy'].unit = 'Jy'
    # tgss['e_flux_jy'].unit = 'Jy'
    
    # if str(tgss['e_ra'].unit) == 'arcsec':
    #     tgss['e_ra'] /= 3600
    #     tgss['e_ra'].unit = 'deg'
    # if str(tgss['e_dec'].unit) == 'arcsec':
    #     tgss['e_dec'] /= 3600
    #     tgss['e_dec'].unit = 'deg'
    
    # tgss.write("tgss_clean.fits", overwrite=True)
    
    #### gleam ####
    # gleam.rename_column("RAJ2000", "ra")
    # gleam.rename_column("DEJ2000", "dec")
    # gleam.rename_column("err_RAJ2000", "e_ra")
    # gleam.rename_column("err_DEJ2000", "e_dec")
    # gleam.rename_column("int_flux", 'flux_jy')
    # gleam.rename_column("err_int_flux", 'e_flux_jy')
    
    # for col in gleam.colnames:
    #     gleam[col].info.description = None
    
    # gleam.write("gleam_300_clean.fits", overwrite=True)
    
    #### nvss ####
    # nvss.rename_column('RAJ2000', 'ra')
    # nvss.rename_column('DEJ2000', 'dec')
    # nvss.rename_column('e_RAJ2000', 'e_ra')
    # nvss.rename_column('e_DEJ2000', 'e_dec')
    # nvss.rename_column('S1_4', 'flux_jy')
    # nvss.rename_column('e_S1_4', 'e_flux_jy')
    
    # nvss['flux_jy'] *= 1e-3
    # nvss['e_flux_jy'] *= 1e-3
    # nvss['flux_jy'].unit = 'Jy'
    # nvss['e_flux_jy'].unit = 'Jy'
    
    # # nvss has weird units
    # if str(nvss['e_ra'].unit) == 's':
    #     nvss['e_ra'] *= 15/3600
    #     nvss['e_ra'].unit = 'deg'
    # if str(nvss['e_dec'].unit) == 'arcsec':
    #     nvss['e_dec'] /= 3600
    #     nvss['e_dec'].unit = 'deg'
    
    # nvss.write("nvss_clean.fits", overwrite=True)
    
    
    #### wenss ####
    # wenss.rename_column('RAJ2000', 'ra')
    # wenss.rename_column('DEJ2000', 'dec')
    # wenss.rename_column('Sint', 'flux_jy')
    # wenss['flux_jy'] = np.array(wenss['flux_jy'], dtype=float) * 1e-3 # WHY is the flux stored as int32?
    # wenss['flux_jy'].unit = 'Jy'
    
    # wenss = wenss[wenss['flux_jy'] > 0]
    
    # wenss['e_ra'] = np.ones_like(wenss['ra'], dtype=float) * 2/3600   # roughly 2" error
    # wenss['e_dec'] = np.ones_like(wenss['dec'], dtype=float) * 2/3600 # roughly 2" error
    # wenss['e_ra'].unit = 'deg'
    # wenss['e_dec'].unit = 'deg'
    
    # wenss['e_flux_jy'] = np.ones_like(wenss['Nse'], dtype=float)
    
    # for i, val in enumerate(wenss['e_flux_jy']):
    #     beam_area = float(wenss['MajAxis'][i]) * float(wenss['MinAxis'][i])
    #     if beam_area > 0: 
    #         wenss['Nse'][i] /= beam_area
        
    # wenss['e_flux_jy'] = np.sqrt(wenss['Nse']**2 + (0.05 * wenss['flux_jy'])**2)
    
    # wenss['e_flux_jy'].unit = 'Jy'
    # wenss['Nse'].unit = 'Jy'
    
    # wenss.write("wenss_clean.fits", overwrite=True)
    
    ###############################################
    #### ensurinig all cleaned catalogs are OK ####
    ###############################################
    racs      = Table.read(os.getcwd()+"/catalogs/racs/racs_clean.fits")
    meerkat   = Table.read(os.getcwd()+"/catalogs/meerkat/meerkat_clean.fits")
    vlssr     = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_clean.fits")
    tgss      = Table.read(os.getcwd()+"/catalogs/tgss/tgss_clean.fits")
    gleam     = Table.read(os.getcwd()+"/catalogs/gleam_300/gleam_300_clean.fits")
    gleam_xgp = Table.read(os.getcwd()+"/catalogs/gleam_x_gp/gleam_x_gp_clean.fits")
    nvss      = Table.read(os.getcwd()+"/catalogs/nvss/nvss_clean.fits")
    wenss     = Table.read(os.getcwd()+"/catalogs/wenss/wenss_clean.fits")
    lofar_dr3 = Table.read(os.getcwd()+"/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits")
    lofar     = Table.read(os.getcwd()+'/catalogs/lofar/lofar_sources_pipeline.fits')
    cygnus    = Table.read(os.getcwd()+'/catalogs/other/cygnus_clean.fits')
    racs_mid  = Table.read(os.getcwd()+"/catalogs/racs/racs_mid_clean.fits")
    racs_high = Table.read(os.getcwd()+"/catalogs/racs/racs_high_clean.fits")
    apertif   = Table.read(os.getcwd()+"/catalogs/apertif/apertif_clean.fits")
    vlass     = Table.read(os.getcwd()+"/catalogs/vlass/vlass_clean.fits")
    txs       = Table.read(os.getcwd()+"/catalogs/txs/txs_clean.fits")
    vcss       = Table.read(os.getcwd()+"/catalogs/vcss/vcss_clean.fits")
    
    cats = [apertif, vlass, txs, vcss, racs_mid, racs_high, meerkat, vlssr, tgss, gleam, gleam_xgp, nvss, wenss, lofar_dr3, lofar, cygnus]
    name = ['apertif', 'vlass', 'txs', 'vcss', 'racs_mid', 'racs_high', 'meerkat', 'vlssr', 'tgss', 'gleam_300', 'gleam_xgp', 'nvss', 'wenss', 'lofar_dr3', 'lofar_pipe', 'cygnus']
    
    for i, cat in enumerate(cats):
        assert "flux_jy" in cat.colnames
        assert "e_flux_jy" in cat.colnames
        assert str(cat["flux_jy"].unit) == 'Jy'
        assert str(cat["e_flux_jy"].unit) == 'Jy'
        
        assert "ra" in cat.colnames
        assert "dec" in cat.colnames
        assert "e_ra" in cat.colnames
        assert "e_dec" in cat.colnames
        
        assert str(cat["e_ra"].unit) == 'deg' or str(cat["e_ra"].unit) == 'None'
        assert str(cat["e_dec"].unit) == 'deg' or str(cat["e_dec"].unit) == 'None'
        
        print(f"PASSED {i+1} / {len(cats)}: {name[i]}")
