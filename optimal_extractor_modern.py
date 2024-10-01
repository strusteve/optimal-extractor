import sys
sys.path.insert(1, '/Users/struanstevenson/Desktop/PhD Research/nested-sampling/nested_sampling')
import numpy as np
from jwst import datamodels
from mh_sampler import metropolis_hastings
from nested_sampler import nested_sampler
import warnings


class OptimalExtractor(object):
    """ Extract optimal spectrum
    
    Attributes
    ----------

    s2d_filename: string
        Path to the 2D spectrum file.
    """

    def __init__(self, s2d_filename):
        self.s2d = datamodels.open(s2d_filename)
        self.s2dsci = self.s2d.data * np.less(0, self.s2d.data)
        self.s2derr = self.s2d.err
        self.s2dwave = self.s2d.wavelength


    def standard_extraction(self):
        """Extract a standard spectrum

        Returns
        -------
        standard_extraction : numpy.ndarray
            1D spectrum calculated via collapsing a 2D spectrum across spatial dimension.
        """

        standard_extraction = np.nansum(self.s2dsci, axis=0)
        standard_extraction[standard_extraction==0] = np.nan
        return standard_extraction
    

    def initial_spatial_profile(self):
        """Calculate an initial spatial profile

        Returns
        -------
        initial_p : numpy.ndarray
            Initial spatial profile of 2D image.
        """
        initial_p = self.s2dsci / np.nansum(self.s2dsci, axis=0)
        return initial_p
    
    
    def gaussian(self, x, params):
        """Gaussian function

        Parameters
        ----------

        x :  float or numpy.ndarray
            Dataset containing x values.

        params : numpy.ndarray
            Array of gaussian parameters (mu, sigma).

        Returns
        -------
        y : float or numpy.ndarray
            Dataset containing y values.
        """
        mu = params[0]
        sigma = params[1]
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (((x - mu)/sigma)**2))
        return y
    
    def log_likelihood(self, data, gen_func, params):
        """ Return the natural logarithm of the Likelihood. Constructed from a single gaussian.

        Parameters
        ----------

        data :  numpy.ndarray
            Dataset containing x, y and y_uncert values arranged in 3 rows.

        gen_func : function
            Generative function used to model the data.

        params : numpy.ndarray
            Free parameters of the generative model.
        """

        xi, yi, yisig = data[0], data[1], data[2]
        log_li = np.log((1/(np.sqrt(2*np.pi*(yisig**2))))) + ((-(yi - gen_func(xi, params))**2) / (2*(yisig**2)))
        log_l_total = np.sum(log_li)

        return(log_l_total)
    
    def smoothed_spatial_profile(self):
        """Get smoothed spatial profile

        Returns
        -------
        smoothed_spatial_profile : numpy.ndarray
            Spatial profile of 2D image.
        """

        initial_p = self.initial_spatial_profile()
        with warnings.catch_warnings(action="ignore"):
            pixel_profile = np.nanmean(initial_p, axis = 1)
            pixel_profile_err = np.nanmean(self.s2derr, axis = 1)

        # Remove nans
        nanmask = ~np.isnan(pixel_profile)
        real_pixel_profile = pixel_profile[nanmask]
        real_pixel_profile_err = pixel_profile_err[nanmask]

        # Run nested sampler
        data = np.array((np.arange(len(real_pixel_profile)), real_pixel_profile, real_pixel_profile_err))
        nest = nested_sampler(100, lambda params: self.log_likelihood(data, self.gaussian, params), -50)
        samples = nest.run_sampler([0, 0], [42, 42])
        mu_fit = np.mean(samples.T[0])
        sig_fit = np.mean(samples.T[1])
        smoothed_pixel_profile = self.gaussian(np.arange(len(pixel_profile), dtype='float'), [mu_fit, sig_fit])
                
        # Enforce positivity and normalisation
        smoothed_pixel_profile = smoothed_pixel_profile * np.less(0, smoothed_pixel_profile)
        smoothed_pixel_profile = smoothed_pixel_profile / np.nansum(smoothed_pixel_profile)

        # Extend for each wavelength (for ease of matrix multiplication)
        smoothed_spatial_profile = np.tile(smoothed_pixel_profile, (len(initial_p[0]), 1)).T

        return smoothed_spatial_profile
    

    def optimal_extraction(self):
        """Optimal extraction main algorithm.

        Returns
        -------
        f_opt : numpy.ndarray
            Optimally extracted 1D spectrum.
        
        var_f_opt : numpy.ndarray
            Variance of the optimally extracted 1D spectrum.
        """

        spatial_profile_image = self.smoothed_spatial_profile()

        with warnings.catch_warnings(action="ignore"):
            f_opt = np.nansum ( ((spatial_profile_image * self.s2dsci) / (self.s2derr**2)), axis=0) / np.nansum( ((spatial_profile_image**2)/(self.s2derr**2)), axis=0)
            var_f_opt = 1 / np.nansum(((spatial_profile_image**2) / (self.s2derr**2)), axis=0)

        return f_opt, var_f_opt