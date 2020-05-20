import itertools
import scipy.integrate
from pytest import approx

import gmw


def test_analytic_normalization():
    betas = [0.1, 0.4, 5, 10, 23]
    gammas = [1, 2, 3, 1.1, 3.5]
    scales = [0.1, 0.5, 1, 2, 10, 100]

    for beta, gamma, scale in itertools.product(betas, gammas, scales):
        wavelet = gmw.MorseWavelet(beta, gamma, norm='analytic')
        cf = wavelet.central_freq(scale)

        assert wavelet.psi_f(cf, scale) == approx(2, abs=1e-10)


def test_energy_normalization():
    betas = [0.1, 0.4, 5, 10, 23]
    gammas = [1, 2, 3, 1.1, 3.5]
    scales = [0.1, 0.5, 1, 2, 10, 20]

    for beta, gamma, scale in itertools.product(betas, gammas, scales):
        wavelet = gmw.MorseWavelet(beta, gamma, norm='energy')

        def wnorm(freq):
            return wavelet.psi_f(freq, scale)**2

        value, _ = scipy.integrate.quad(wnorm, 0, float('inf'), epsabs=1e-10)

        assert value == approx(1, abs=1e-10)
