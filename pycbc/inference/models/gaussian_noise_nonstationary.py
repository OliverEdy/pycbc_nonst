"""This module provides model classes that assume the noise is Gaussian and non-stationary.
The non-stationary noise covariance needs to be calculated separately and loaded in.
"""

import logging
# from ConfigParser import NoSectionError, NoOptionError
from argparse import ArgumentParser

import numpy
from .gaussian_noise import GaussianNoise

# # Parse command line arguments
# parser = ArgumentParser()
# parser.add_argument('--Sigma_Inverted', default=None, type=float)
# args = parser.parse_args()
# Sigma_Inverted_Loaded = args.threshold
Sigma_Inverted_Loaded = numpy.identity(10)

class NonStationaryGaussianNoise(GaussianNoise):

    name = 'gaussian_noise_nonstationary'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, nonstationary_noise_covariance=Sigma_Inverted_Loaded, **kwargs):
        # set up the boiler-plate attributes
        super(NonStationaryGaussianNoise, self).__init__(variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        # load the non-stationary noise covariance
        self.nonstationary_noise_covariance = nonstationary_noise_covariance
        # use the non-stationary noise covariance for the loglikelihood
        self._loglikelihood = self._loglikelihood_nonst

#     @property
#     def _extra_stats(self):
#         """Adds ``loglr`` and ``lognl`` to the ``default_stats``."""
#         return ['loglr', 'lognl']

    def _loglikelihood_nonst(self):
        # Sigma_inv = self.nonstationary_noise_covariance
        
        det_logls = {}
        for (det, d) in self._data.items():
            n = numpy.mat(d).T
            Sigma_inv = numpy.identity(n.shape[0])
            loglikelihood = n.H.dot(Sigma_inv).dot(n)
            det_logls[det] = numpy.real(loglikelihood).tolist()[0][0]
                
        logl = sum(det_logls.values())
        # setattr(self._current_stats, 'loglikelihood', logl)
        self._current_stats.loglikelihood = logl
        return logl