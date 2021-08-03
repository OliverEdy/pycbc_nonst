"""This module provides model classes that assume the noise is Gaussian and non-stationary.
The non-stationary noise covariance needs to be calculated separately and loaded in.
"""

import logging
# from ConfigParser import NoSectionError, NoOptionError
from argparse import ArgumentParser

import numpy
import scipy
from scipy.linalg import dft as Make_DFT
import pycbc
from pycbc.inference.models import GaussianNoise
from .gaussian_noise import GaussianNoise

# # Parse command line arguments
# parser = ArgumentParser()
# parser.add_argument('--Sigma_Inverted', default=None, type=float)
# args = parser.parse_args()
# Sigma_Inverted_Loaded = args.threshold
Sigma_Inverted_Loaded = numpy.identity(10)

'''
Woodbury functions
'''

def Simplifying_S1_and_S2_for_Woodbury(A, C, V):
    A_sqrt = numpy.sqrt(A)
    A_sqrt_inv = numpy.mat(numpy.diag(1./numpy.diag(A_sqrt))) # Quick way to invert a diagonal matrix
    C_sqrt = numpy.sqrt(C)
    V_new = C_sqrt @ V @ A_sqrt_inv
    return A_sqrt_inv, V_new

def SVD_for_Woodbury(V, num_dec_places=5):
    u, s, vT = scipy.linalg.svd(V)
    s = s[numpy.around(s, num_dec_places) > 0]
    m = s.shape[0]
    vT = vT[0:m, :]
    V_mxn = numpy.diag(s) @ vT
    return V_mxn, m

def Woodbury(A, U, V, m):
    A_inv = numpy.diag(1./numpy.diag(A))
    B_inv = numpy.linalg.inv(numpy.identity(m) + (V * A_inv) @ U)
    return A_inv - (A_inv * U @ B_inv @ V * A_inv)

'''
Non-Stationary noise model
'''

class NonStationaryGaussianNoise(GaussianNoise):

    name = 'gaussian_noise_nonstationary'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, **kwargs):
        # set up the boiler-plate attributes
        super(NonStationaryGaussianNoise, self).__init__(variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
#         # load the non-stationary noise covariance
#         self.nonstationary_noise_covariance = nonstationary_noise_covariance
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
            N = n.shape[0]
            U = numpy.mat(Make_DFT(N))
            
            B_t = numpy.mat(numpy.diag(modulation))
            U_inv = U.H

            V = U @ B_t @ U_inv
        
            S1 = numpy.identity(N)
            S2 = numpy.identity(N)
            del U, U_inv, B_t

            S1_sqrt_inv, V_new = Simplifying_S1_and_S2_for_Woodbury(S1, S2, V)
        
            V_mxn, m = SVD_for_Woodbury(V_new)    
            
            V_mxn_H_times_n = V_mxn @ n
            loglikelihood = n.H @ n - V_mxn_H_times_n.H @ V_mxn_H_times_n
            det_logls[det] = numpy.real(loglikelihood).tolist()[0][0]
                
        logl = sum(det_logls.values())
        # setattr(self._current_stats, 'loglikelihood', logl)
        self._current_stats.loglikelihood = logl
        return logl