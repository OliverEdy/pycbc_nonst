"""This module provides model classes that assume the noise is Gaussian and non-stationary.
The non-stationary noise covariance needs to be calculated separately and loaded in.
"""

import numpy
from pycbc.waveform import (NoWaveformError, FailedWaveformError)
from .gaussian_noise import GaussianNoise

try:
    V_mxn = numpy.load('./V_mxn.npy', allow_pickle=True)
except:
    pass

'''
Non-Stationary noise model
'''

class NonStationaryGaussianNoise(GaussianNoise):

    name = 'gaussian_noise_nonstationary'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, V_mxn=None, **kwargs):
        # set up the boiler-plate attributes
        super().__init__(variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        self.V_mxn = V_mxn

        
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
            slc = slice(kmin, kmax)
            d = self._whitened_data[det][slc]
            lognorm = self.det_lognorm(det)
            
            if self.V_mxn is None:
                V_mxn = numpy.zeros(self._whitened_data[det].shape)
            else:
                V_mxn = self.V_mxn
            
            V_mxn_times_d = numpy.matmul(V_mxn[:, slc], d)[:, numpy.newaxis]
            dd = d.inner(d) - numpy.dot(V_mxn_times_d.conj().T, V_mxn_times_d)[0][0]
            
            lognl = lognorm - 0.5 * dd.real
            self._det_lognls[det] = lognl
            return self._det_lognls[det]
        

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
                if self.V_mxn is None:
                    V_mxn = numpy.zeros(self._whitened_data[det].shape)
                else:
                    V_mxn = self.V_mxn
                
                # whiten the waveform
                h[self._kmin[det]:kmax] *= self._weight[det][slc]
                h = h[slc]
                d = self._whitened_data[det][slc]
                
                # the inner products
                V_mxn_times_d = numpy.matmul(V_mxn[:, slc], d)[:, numpy.newaxis]
                V_mxn_times_h = numpy.matmul(V_mxn[:, slc], h)[:, numpy.newaxis]
                cplx_hd = d.inner(h) - numpy.dot(V_mxn_times_h.conj().T, V_mxn_times_d)[0][0]
                
                hh = (h.inner(h) - numpy.dot(V_mxn_times_h.conj().T, V_mxn_times_h)[0][0]).real
                
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