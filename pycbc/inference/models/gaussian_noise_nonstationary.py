"""This module provides model classes that assume the noise is Gaussian and non-stationary.
The non-stationary noise covariance needs to be calculated separately and loaded in.
"""

import logging
from ConfigParser import NoSectionError, NoOptionError

import numpy
from .gaussian_noise import GaussianNoise

class GaussianNoise_NonStationary(GaussianNoise):

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, nonstationary_noise_covariance=None, **kwargs):
        # set up the boiler-plate attributes
        super(GaussianNoise_NonStationary, self).__init__(variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        self.nonstationary_noise_covariance = nonstationary_noise_covariance
        self._loglikelihood = self._loglikelihood_nonst

    def _loglikelihood_nonst(self):
        Sigma_inv = self.nonstationary_noise_covariance
        
        det_logls = {}
        for (det, d) in self._data.items():
            n = numpy.mat(d).T
            loglikelihood = n.H.dot(Sigma_inv).dot(n)
            det_logls[det] = numpy.real(loglikelihood).tolist()[0][0]
                
        logl = sum(det_logls.values())
        return logl