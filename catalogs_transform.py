import os
from astropy.table import Table, Column
import bdsf
import glob
import numpy as np

# get calagogs
# racs      = Table.read(os.getcwd()+"/catalogs/racs/RACS_DR1_Sources_GalacticRegion_v2021_08.xml")
# meerkat   = Table.read(os.getcwd()+"/catalogs/meerkat/smgps_moment0_5beam_5sigma_510599row_compact_source_catalogue.fits")
# vlssr     = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_full.csv")
# tgss      = Table.read(os.getcwd()+"/catalogs/tgss/TGSSADR1_7sigma_catalog.fits")
# gleam     = Table.read(os.getcwd()+"/catalogs/gleam_300/GLEAM300_source_catalogue.fits")
# gleam_xgp = Table.read(os.getcwd()+"/catalogs/gleam_x_gp/gleam_x_gp.fit")
# nvss      = Table.read(os.getcwd()+"/catalogs/nvss/NVSS.fits")
# wenss     = Table.read(os.getcwd()+"/catalogs/wenss/WENSS.fits")
# lofar_dr3 = Table.read(os.getcwd()+"/catalogs/lofar/LoTSS_DR3_v1.0.srl.fits")
# lofar     = Table.read(os.getcwd()+'/catalogs/lofar/lofar_sources_pipeline.fits')

# cygnus = Table.read(os.getcwd()+'/catalogs/other/cygnus_sources.fits')

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

#### racs ####
# racs['flux_jy'] = racs['Total_flux_Source'] * 1e-3
# racs['e_flux_jy'] = racs['E_Total_flux_Source'] * 1e-3
# racs['flux_jy'].unit = 'Jy'
# racs['e_flux_jy'].unit = 'Jy'
# racs.rename_column("RA", "ra")
# racs.rename_column("Dec", "dec")
# racs.rename_column("E_RA", "e_ra")
# racs.rename_column("E_Dec", "e_dec")

# if str(racs['e_ra'].unit) == 'arcsec':
#     racs['e_ra'] /= 3600
#     racs['e_ra'].unit = 'deg'
# if str(racs['e_dec'].unit) == 'arcsec':
#     racs['e_dec'] /= 3600
#     racs['e_dec'].unit = 'deg'

# racs_out = Table()
# for col in racs.colnames:
#     data = np.array(racs[col])  # strips Quantity/mixin type
#     unit = str(racs[col].unit) if racs[col].unit else None
#     racs_out[col] = Column(data, unit=unit)

# racs_out.write("racs_clean.fits", overwrite=True)

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
# racs      = Table.read(os.getcwd()+"/catalogs/racs/racs_clean.fits")
# meerkat   = Table.read(os.getcwd()+"/catalogs/meerkat/meerkat_clean.fits")
# vlssr     = Table.read(os.getcwd()+"/catalogs/vlssr/vlssr_clean.fits")
# tgss      = Table.read(os.getcwd()+"/catalogs/tgss/tgss_clean.fits")
# gleam     = Table.read(os.getcwd()+"/catalogs/gleam_300/gleam_300_clean.fits")
# gleam_xgp = Table.read(os.getcwd()+"/catalogs/gleam_x_gp/gleam_x_gp_clean.fits")
# nvss      = Table.read(os.getcwd()+"/catalogs/nvss/nvss_clean.fits")
# wenss     = Table.read(os.getcwd()+"/catalogs/wenss/wenss_clean.fits")
# lofar_dr3 = Table.read(os.getcwd()+"/catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits")
# lofar = Table.read(os.getcwd()+'/catalogs/lofar/lofar_sources_pipeline.fits')
# cygnus = Table.read(os.getcwd()+'/catalogs/other/cygnus_clean.fits')

# cats = [racs, meerkat, vlssr, tgss, gleam, gleam_xgp, nvss, wenss, lofar_dr3, lofar, cygnus]
# name = ['racs', 'meerkat', 'vlssr', 'tgss', 'gleam_300', 'gleam_xgp', 'nvss', 'wenss', 'lofar_dr3', 'lofar_pipe', 'cygnus']

# for i, cat in enumerate(cats):
#     assert "flux_jy" in cat.colnames
#     assert "e_flux_jy" in cat.colnames
#     assert str(cat["flux_jy"].unit) == 'Jy'
#     assert str(cat["e_flux_jy"].unit) == 'Jy'
    
#     assert "ra" in cat.colnames
#     assert "dec" in cat.colnames
#     assert "e_ra" in cat.colnames
#     assert "e_dec" in cat.colnames
    
#     assert str(cat["e_ra"].unit) == 'deg' or str(cat["e_ra"].unit) == 'None'
#     assert str(cat["e_dec"].unit) == 'deg' or str(cat["e_dec"].unit) == 'None'
    
#     print(f"PASSED {i+1} / {len(cats)}: {name[i]}")
