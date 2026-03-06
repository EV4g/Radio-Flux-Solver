
SMGPS Compact Source Catalogue (Mutale+, 2025)
==================================================================================================================================

Description:
	A catalogue of 510,599 compact radio sources (area < 5 synthesized beams) detected in the SARAO MeerKAT Galactic Plane Survey
	above a signal-to-noise threshold of 5

----------------------------------------------------------------------------------------------------------------------------------
Col. #	|  Col. Name	| Units	   | Description
----------------------------------------------------------------------------------------------------------------------------------	
1 	|  csc_id 	| --- 	   | Catalogue source identification number
2 	|  Name 	| --- 	   | Galactic source name
3 	|  IAUName 	| --- 	   | IAU designation of the source name
4 	|  tileName	| --- 	   | File identifier of mosaic from which the source was extracted
5 	|  island	| --- 	   | A group of 1 or more Gaussians identified as pixels with contiguous source emission
6 	|  source 	| --- 	   | A single Gaussian component
7 	|  background	| mJy/beam | background flux density
8 	|  local_rms	| mJy/beam | rms noise around the immediate vicinity of the source
9 	|  lon 		| deg 	   | Galactic longitude of the centroid position of the source
10	|  err_lon 	| deg 	   | Source-extraction fitting error on Galactic longitude
11	|  lat 		| deg 	   | Galactic latitude of the centroid position of the source
12	|  err_lat 	| deg 	   | Source-extraction fitting error on Galactic latitude
13	|  ra 		| deg 	   | Right Ascension of the centroid position of the source
14	|  dec 		| deg 	   | Declination of the centroid position of the source
15	|  peak_flux 	| mJy/beam | Peak flux density
16	|  err_peak_flux| mJy/beam | Source-extraction fitting error on peak flux density
17	|  int_flux 	| mJy 	   | Integrated flux density (calculated as a/b/peak_flux)
18	|  err_int_flux | mJy 	   | Source-extraction fitting error on integrated flux density
19	|  a 		| arcsec   | Fitted major axis
20	|  err_a 	| arcsec   | Error on fitted major axis
21	|  b 		| arcsec   | Fitted minor axis
22	|  err_b 	| arcsec   | Error on fitted minor axis
23	|  pa 		| deg 	   | Fitted position angle
24	|  err_pa 	| deg 	   | Error on fitted position angle
25	|  flags 	| --- 	   | Fitting flags (see main text for description)
26	|  snr 		| --- 	   | Signal-to-noise ratio (calculated as peak_flux/local_rms)
27	|  area 	| arcsec2  | Source area (calculated as pi.a.b/4ln2)
28	|  nbs		| ---	   | Near bright source flag (yes = within 0.5 degrees of a bright source)
----------------------------------------------------------------------------------------------------------------------------------