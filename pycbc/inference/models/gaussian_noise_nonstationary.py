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
# from pycbc.inference.models import GaussianNoise
from .gaussian_noise import GaussianNoise

# # Parse command line arguments
# parser = ArgumentParser()
# parser.add_argument('--Sigma_Inverted', default=None, type=float)
# args = parser.parse_args()
# Sigma_Inverted_Loaded = args.threshold

# '''
# Woodbury functions
# '''
# 
# def Simplifying_S1_and_S2_for_Woodbury(A, C, V):
#     A_sqrt = numpy.sqrt(A)
#     A_sqrt_inv = numpy.mat(numpy.diag(1./numpy.diag(A_sqrt))) # Quick way to invert a diagonal matrix
#     C_sqrt = numpy.sqrt(C)
#     V_new = C_sqrt @ V @ A_sqrt_inv
#     return A_sqrt_inv, V_new
# 
# def SVD_for_Woodbury(V, num_dec_places=5):
#     u, s, vT = scipy.linalg.svd(V)
#     s = s[numpy.around(s, num_dec_places) > 0]
#     m = s.shape[0]
#     vT = vT[0:m, :]
#     V_mxn = numpy.diag(s) @ vT
#     return V_mxn, m
# 
# def Woodbury(A, U, V, m):
#     A_inv = numpy.diag(1./numpy.diag(A))
#     B_inv = numpy.linalg.inv(numpy.identity(m) + (V * A_inv) @ U)
#     return A_inv - (A_inv * U @ B_inv @ V * A_inv)

'''
Non-Stationary noise model
'''

class NonStationaryGaussianNoise(BaseGaussianNoise):

    name = 'gaussian_noise_nonstationary'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, **kwargs):
        # set up the boiler-plate attributes
        super(NonStationaryGaussianNoise, self).__init__(variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        # create the waveform generator
        self.waveform_generator = create_waveform_generator(
            self.variable_params, self.data,
            waveform_transforms=self.waveform_transforms,
            recalibration=self.recalibration,
            gates=self.gates, **self.static_params)
#         # load the non-stationary noise covariance
#         self.nonstationary_noise_covariance = nonstationary_noise_covariance
        # load the resized V matrix
        self.V_mxn = numpy.mat(numpy.load('./V_mxn.npy', allow_pickle=True))
        self.S1_sqrt_inv = numpy.mat(numpy.load('./S1_sqrt_inv.npy', allow_pickle=True))

#     @property
#     def _extra_stats(self):
#         """Adds ``loglr`` and ``lognl`` to the ``default_stats``."""
#         return ['loglr', 'lognl']

    def det_lognl(self, det):
        r"""Returns the log likelihood of the noise in the given detector:

        .. math::

            \log p(d_i|n_i) = \log \alpha_i -
                \frac{1}{2} \left<d_i | d_i\right>.


        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        float :
            The log likelihood of the noise in the requested detector.
        """
        try:
            return self._det_lognls[det]
        except KeyError:
            # hasn't been calculated yet; calculate & store
            kmin = self._kmin[det]
            kmax = self._kmax[det]
            d = self._whitened_data[det]
            lognorm = self.det_lognorm(det)
            
            S1_times_d = numpy.multiply(numpy.diag(S1_sqrt_inv)[:,None], numpy.mat(d[kmin:kmax]).T)
            V_mxn_times_d = V_mxn @ S1_times_d
            dd_as_a_matrix = S1_times_d.H @ S1_times_d - V_mxn_times_d.H @ V_mxn_times_d
            dd = numpy.real(dd_as_a_matrix).tolist()[0][0]
            
            lognl = lognorm - 0.5 * dd#d[kmin:kmax].inner(d[kmin:kmax]).real
            self._det_lognls[det] = lognl
            return self._det_lognls[det]

    def _lognl(self):
        """Computes the log likelihood assuming the data is noise.

        Since this is a constant for Gaussian noise, this is only computed once
        then stored.
        """
        return sum(self.det_lognl(det) for det in self._data)

    def _loglikelihood(self):
        r"""Computes the log likelihood of the paramaters,

        .. math::

            \log p(d|\Theta, h) = \log \alpha -\frac{1}{2}\sum_i
                \left<d_i - h_i(\Theta) | d_i - h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood evaluated at the given point.
        """
        # since the loglr has fewer terms, we'll call that, then just add
        # back the noise term that canceled in the log likelihood ratio
        return self.loglr + self.lognl

    @property
    def _extra_stats(self):
        """Adds ``loglr``, plus ``cplx_loglr`` and ``optimal_snrsq`` in each
        detector."""
        return ['loglr'] + \
               ['{}_cplx_loglr'.format(det) for det in self._data] + \
               ['{}_optimal_snrsq'.format(det) for det in self._data]

    def _nowaveform_loglr(self):
        """Convenience function to set loglr values if no waveform generated.
        """
        for det in self._data:
            setattr(self._current_stats, 'loglikelihood', -numpy.inf)
            setattr(self._current_stats, '{}_cplx_loglr'.format(det),
                    -numpy.inf)
            # snr can't be < 0 by definition, so return 0
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det), 0.)
        return -numpy.inf
        
#     def _loglikelihood_nonst(self):
#         V_mxn = self.V_mxn
#         S1_sqrt_inv = self.S1_sqrt_inv
#         # Sigma_inv = self.nonstationary_noise_covariance
#         
#         det_logls = {}
#         for (det, d) in self._data.items():
#             n = numpy.mat(d).T
# 
# #             N = n.shape[0]
# #             U = numpy.mat(Make_DFT(N))
# # 
# #             B_t = numpy.mat(numpy.diag(d))#modulation))
# #             U_inv = U.H
# # 
# #             V = U @ B_t @ U_inv
# # 
# #             S1 = numpy.identity(N)
# #             S2 = numpy.identity(N)
# #             del U, U_inv, B_t
# # 
# #             S1_sqrt_inv, V_new = Simplifying_S1_and_S2_for_Woodbury(S1, S2, V)
# # 
# #             V_mxn, m = SVD_for_Woodbury(V_new)
# 
#             S1_times_n = numpy.multiply(numpy.diag(S1_sqrt_inv)[:,None], n)
#             V_mxn_times_n = V_mxn @ S1_times_n
#             loglikelihood = S1_times_n.H @ S1_times_n - V_mxn_times_n.H @ V_mxn_times_n
#             det_logls[det] = numpy.real(loglikelihood).tolist()[0][0]
#                 
#         logl = sum(det_logls.values())
#         # setattr(self._current_stats, 'loglikelihood', logl)
#         self._current_stats.loglikelihood = logl
#         return logl

    def _loglr(self):
        r"""Computes the log likelihood ratio,

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood ratio.
        """
        params = self.current_params
        try:
            wfs = self.waveform_generator.generate(**params)
        except NoWaveformError:
            return self._nowaveform_loglr()
        except FailedWaveformError as e:
            if self.ignore_failed_waveforms:
                return self._nowaveform_loglr()
            else:
                raise e
        lr = 0.
        for det, h in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(len(h), self._kmax[det])
            if self._kmin[det] >= kmax:
                # if the waveform terminates before the filtering low frequency
                # cutoff, then the loglr is just 0 for this detector
                cplx_hd = 0j
                hh = 0.
            else:
                slc = slice(self._kmin[det], kmax)
                # whiten the waveform
                h[self._kmin[det]:kmax] *= self._weight[det][slc]
                h_mat = numpy.mat(h).T
                d_mat = numpy.mat(self._whitened_data[det][slc]).T
                
                # the inner products
                # cplx_hd = self._whitened_data[det][slc].inner(h[slc])  # <h, d>
                S1_times_d = numpy.multiply(numpy.diag(S1_sqrt_inv)[:,None], d_mat)
                S1_times_h = numpy.multiply(numpy.diag(S1_sqrt_inv)[:,None], h_mat)
                V_mxn_times_d = V_mxn @ S1_times_d
                V_mxn_times_h = V_mxn @ S1_times_h
                dh_as_a_matrix = S1_times_h.H @ S1_times_d - V_mxn_times_h.H @ V_mxn_times_d
                cplx_dh = dh_as_a_matrix.tolist()[0][0]
                
                # hh = h[slc].inner(h[slc]).real  # < h, h>
                hh_as_a_matrix = S1_times_h.H @ S1_times_h - V_mxn_times_h.H @ V_mxn_times_h
                cplx_hh = numpy.real(hh_as_a_matrix).tolist()[0][0]
                
            cplx_loglr = cplx_hd - 0.5*hh
            # store
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det), hh)
            setattr(self._current_stats, '{}_cplx_loglr'.format(det),
                    cplx_loglr)
            lr += cplx_loglr.real
        # also store the loglikelihood, to ensure it is populated in the
        # current stats even if loglikelihood is never called
        self._current_stats.loglikelihood = lr + self.lognl
        return float(lr)

    def det_cplx_loglr(self, det):
        """Returns the complex log likelihood ratio in the given detector.

        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        complex float :
            The complex log likelihood ratio.
        """
        # try to get it from current stats
        try:
            return getattr(self._current_stats, '{}_cplx_loglr'.format(det))
        except AttributeError:
            # hasn't been calculated yet; call loglr to do so
            self._loglr()
            # now try returning again
            return getattr(self._current_stats, '{}_cplx_loglr'.format(det))

    def det_optimal_snrsq(self, det):
        """Returns the optimal SNR squared in the given detector.

        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        float :
            The opimtal SNR squared.
        """
        # try to get it from current stats
        try:
            return getattr(self._current_stats, '{}_optimal_snrsq'.format(det))
        except AttributeError:
            # hasn't been calculated yet; call loglr to do so
            self._loglr()
            # now try returning again
            return getattr(self._current_stats, '{}_optimal_snrsq'.format(det))