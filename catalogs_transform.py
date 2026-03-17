import os
from astropy.table import Table
import bdsf
import glob

# get calagogs
# racs    = Table.read(os.getcwd()+"/catalogs/racs/RACS_DR1_Sources_GalacticRegion_v2021_08.xml")
# meerkat = Table.read(os.getcwd()+"/catalogs/meerkat/smgps_moment0_5beam_5sigma_510599row_compact_source_catalogue.csv")
# vlssr   = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_full.csv")
# tgss    = Table.read(os.getcwd()+"/catalogs/tgss/TGSSADR1_7sigma_catalog.fits")
#gleam     = Table.read(os.getcwd()+"/catalogs/gleam/GLEAM300_source_catalogue.fits")


# from astroquery.vizier import Vizier
# v = Vizier(row_limit=-1)
# catalogs = v.get_catalogs("VIII/97")
# vlssr = catalogs[0]
# vlssr.write("vlssr_full.csv", format="ascii.csv", overwrite=True)

#### lofar ####
#lofar_files = np.sort(glob.glob(os.getcwd()+"/data/lofar/*.fits"))[0]
# img = bdsf.process_image(
#     lofar_files,
#     thresh_isl=3.0,       # island threshold (sigma)
#     thresh_pix=5.0,       # peak detection threshold (sigma)
#     rms_box=(100, 25),    # (box_size, step_size) for rms map; tune to your image
#     beam=(get_beam_size(lofar_files)),  # (maj_deg, min_deg, PA)
# )
# img.write_catalog(outfile="lofar_sources_pipeline.fits", format="fits", catalog_type="srl", clobber=True)


#### racs ####
# racs['flux_jy'] = racs['Total_flux_Source'] * 1e-3
# racs['e_flux_jy'] = racs['E_Total_flux_Source'] * 1e-3
# racs.rename_column("RA", "ra")
# racs.rename_column("Dec", "dec")
# racs_out = racs["Source_Name", "ra", "dec", "flux_jy", "e_flux_jy"]
# racs_out.write("racs_clean.csv", format="ascii.csv", overwrite=True)

#### meerkat ####
# meerkat['flux_jy'] = meerkat['int_flux'] * 1e-3
# meerkat['e_flux_jy'] = meerkat['err_int_flux'] * 1e-3
# meerkat_out = meerkat["csc_id", "ra", "dec", "snr", "flux_jy", "e_flux_jy"]
# meerkat_out.write("meerkat_clean.csv", format="ascii.csv", overwrite=True)

#### vlssr ####
# DEG_TO_ARCSEC = 3600.0
# VLSSR_BEAM_ARCSEC = 80.0  # circular restoring beam

# src_maj = vlssr["MajAx"] * DEG_TO_ARCSEC
# src_min = vlssr["MinAx"] * DEG_TO_ARCSEC

# vlssr['flux_jy'] = vlssr["Sp"] * (src_maj * src_min) / (VLSSR_BEAM_ARCSEC ** 2)
# vlssr['e_flux_jy'] = vlssr["e_Sp"] * (src_maj * src_min) / (VLSSR_BEAM_ARCSEC ** 2)

# vlssr.rename_column("ra_deg", "ra")
# vlssr.rename_column("dec_deg", "dec")

# vlssr_out = vlssr["ra", "dec", "flux_jy", 'e_flux_jy']
# vlssr_out.write("vlssr_clean.csv", format="ascii.csv", overwrite=True)

#### tgss ####
# tgss.rename_column('RA', 'ra')
# tgss.rename_column('DEC', 'dec')
# tgss.rename_column('Total_flux', 'flux_jy')
# tgss.rename_column('E_Total_flux', 'e_flux_jy')

# tgss['flux_jy'] *= 1e-3
# tgss['e_flux_jy'] *= 1e-3

# tgss.write("tgss_clean.fits", overwrite=True)

#### gleam ####
# gleam.rename_column("RAJ2000", "ra")
# gleam.rename_column("DEJ2000", "dec")
# gleam.rename_column("int_flux", 'flux_jy')
# gleam.rename_column("err_int_flux", 'e_flux_jy')

# for col in gleam.colnames:
#     gleam[col].info.description = None

# gleam.write("gleam_300_clean.fits", overwrite=True)