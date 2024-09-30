import os
import glob
import numpy as np
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch, AsinhStretch
import matplotlib.pyplot as plt
import corner
from jwst import datamodels
from optimal_extractor_modern import OptimalExtractor
os.environ["CRDS_PATH"] = "/Users/struanstevenson/crds_cache"
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'


# good spectra [8(same), 9(worse), 13(much better), 17(better), 19(worse), 20(same), 23(much worse), 33(same)]
ind=9

############################


def runplot(ind):

    # Get 2D spectra data
    s2d_list = [f for f in sorted(glob.glob('pipelinefinal/*_s2d.fits'))]
    s2d_file = s2d_list[ind]

    # Run optimal extractor
    extraction = OptimalExtractor(s2d_file)
    f_std = extraction.standard_extraction()
    spat_profile = extraction.smoothed_spatial_profile()
    f_opt, var_f_opt = extraction.optimal_extraction()

    # Plot 2D spectrum
    s2d = datamodels.open(s2d_file)
    s2dsci = s2d.data * np.less(0, s2d.data)
    s2dwave = s2d.wavelength
    norm = ImageNormalize(s2dsci, interval=ManualInterval(vmin=0, vmax=0.1), stretch=LinearStretch())
    wcsobj = s2d.meta.wcs
    y, x = np.mgrid[:s2dsci.shape[0], : s2dsci.shape[1]]
    det2sky = wcsobj.get_transform('detector', 'world')
    s2d3ra, s2d3dec, s2d3wave = det2sky(x, y)
    s2d3waves = s2d3wave[0, :]
    xtint = np.arange(100, s2dsci.shape[1], 100)
    xtlab = np.round(s2d3waves[xtint], 2)
    fig, ax = plt.subplots()
    ax.set_title('2D spectrum')
    ax.imshow(s2dsci, norm=norm)
    ax.set_xticks(xtint, xtlab)
    ax.set_xlabel('wavelength (microns)')
    ax.set_ylabel('pixel row')
    plt.show()

    # Compare spatial profiles
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(s2dwave[0], spat_profile.T, color='black')
    ax2.plot(spat_profile, color='black')
    ax1.set_title('Pixel row spatial profile')
    ax2.set_title('Wavelength spatial profile')
    plt.show()

    # Compare 1D spectra
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(s2dwave[0], f_std, label='Standard Extraction', color='black', lw=1)
    ax2.plot(s2dwave[0], f_opt, label='Optimal Extraction', color='black', lw=1)
    ax2.plot(s2dwave[0], np.sqrt(var_f_opt), color='black')
    ax2.set_xlabel('Wavelength (microns)')
    x1dlist = [f for f in sorted(glob.glob('pipelinefinal/*_x1d.fits'))]
    x1d_file = x1dlist[ind]
    x1d = datamodels.open(x1d_file)
    x1dwave = x1d.spec[0].spec_table.WAVELENGTH
    x1dflux = x1d.spec[0].spec_table.FLUX
    ax3.plot(x1dwave, x1dflux, lw=1, color='black', label='JWST Pipeline')
    ax3.set_xlabel('wavelength (microns)')
    ax3.set_ylabel('flux (Jy)')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.subplots_adjust(hspace=0)
    plt.show()

runplot(ind)