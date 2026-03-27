"""
reconvolve_fits.py
------------------
Reconvolve a LOFAR DDF image (Jy/beam) from its native clean-beam
resolution to an arbitrary target beam.

Target: 53" x 38", PA = 56 deg (East-of-North convention).
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
from astropy.convolution import convolve_fft
from radio_beam import Beam

# ── 1. Configuration ─────────────────────────────────────────────────────────
INPUT_FITS  = "/home/floris/Documents/PhD/Galactic plane/LOFAR_and_MeerKAT/data/other/reimage_full.app.restored.fits"
OUTPUT_FITS = "lofar_reconvolved_53x38_pa56.fits"

TARGET_BMAJ = 53.0 * u.arcsec
TARGET_BMIN = 38.0 * u.arcsec
TARGET_BPA  = 56.0 * u.deg        # Position angle East-of-North

# DDF used Output-RestoringBeam = 6.0 (circular), but never wrote BMAJ/BMIN/BPA
# → define manually from the HISTORY
CURRENT_BEAM = Beam(major=6.0 * u.arcsec,
                    minor=6.0 * u.arcsec,
                    pa=0.0    * u.deg)

TARGET_BEAM  = Beam(major=53.0 * u.arcsec,
                    minor=38.0 * u.arcsec,
                    pa=56.0   * u.deg)

# ── Load FITS ─────────────────────────────────────────────────────────────────
with fits.open(INPUT_FITS, memmap=True) as hdul:   # memmap avoids loading all 4D at once
    header = hdul[0].header.copy()
    data   = hdul[0].data.astype(np.float64)       # shape: (1, 1, 19845, 19845)

# Squeeze [stokes, freq, dec, ra] → 2D
data2d = np.squeeze(data)    # (19845, 19845)
print(f"Working array shape : {data2d.shape}")

# ── Pixel scale (directly from CDELT, no WCS object needed) ───────────────────
# CDELT1 = -0.000416... deg/px  →  1.5 arcsec/px
pixscale = abs(header['CDELT1']) * 3600 * u.arcsec
print(f"Pixel scale         : {pixscale:.4f}")

# ── Build convolution kernel ───────────────────────────────────────────────────
# Solve: TARGET = CURRENT ⊗ KERNEL
kernel_beam = TARGET_BEAM.deconvolve(CURRENT_BEAM)
print(f"Current beam        : {CURRENT_BEAM}")
print(f"Target beam         : {TARGET_BEAM}")
print(f"Convolution kernel  : {kernel_beam}")

# as_kernel() returns a normalised (sum=1) Gaussian2DKernel in pixel space
kernel = kernel_beam.as_kernel(pixscale)
print(f"Kernel array size   : {kernel.shape}")

# ── Flux rescaling: Jy/beam → convolve → Jy/beam ─────────────────────────────
# Beam solid angle in pixel² = (π / 4ln2) * BMAJ_pix * BMIN_pix
def beam_area_pix2(beam, pixscale):
    return (np.pi / (4 * np.log(2))
            * (beam.major.to(u.arcsec) / pixscale).decompose().value
            * (beam.minor.to(u.arcsec) / pixscale).decompose().value)

old_area = beam_area_pix2(CURRENT_BEAM, pixscale)
new_area = beam_area_pix2(TARGET_BEAM,  pixscale)
print(f"\nOld beam area : {old_area:.2f} pix²  ({CURRENT_BEAM.major.to(u.arcsec):.1f} circ.)")
print(f"New beam area : {new_area:.2f} pix²")

# ── Convolution ───────────────────────────────────────────────────────────────
# Step 1: Jy/beam → Jy/pixel
data_jy_pix = data2d / old_area

# Step 2: FFT convolve (kernel sums to 1 → flux conserved per pixel)
print("\nConvolving... (this may take a few minutes for 19845² pixels)")
data_conv = convolve_fft(
    data_jy_pix,
    kernel,
    normalize_kernel=True,
    nan_treatment='fill',
    fill_value=0.0,
    allow_huge=True,          # required for ~20k² images
    fft_pad=False,            # image is already huge; skip zero-padding
)

# Step 3: Jy/pixel → Jy/new_beam
data_out = (data_conv * new_area).reshape(data.shape).astype(np.float32)

# ── Update header ─────────────────────────────────────────────────────────────
header['BMAJ'] = TARGET_BEAM.major.to(u.deg).value
header['BMIN'] = TARGET_BEAM.minor.to(u.deg).value
header['BPA']  = TARGET_BEAM.pa.to(u.deg).value
header['HISTORY'] = f"Reconvolved from 6\" circular to {TARGET_BEAM.major.to(u.arcsec):.0f}\" x {TARGET_BEAM.minor.to(u.arcsec):.0f}\" PA={TARGET_BEAM.pa.to(u.deg):.0f} using radio_beam + astropy.convolve_fft"

# ── Write output ──────────────────────────────────────────────────────────────
fits.writeto(OUTPUT_FITS, data_out, header, overwrite=True)
print(f"\n✓ Written: {OUTPUT_FITS}")
print(f"  BMAJ={header['BMAJ']*3600:.1f}\"  BMIN={header['BMIN']*3600:.1f}\"  BPA={header['BPA']:.1f}°")