import numpy as np
from scipy import fft
from scipy.special import loggamma, eval_genlaguerre


class MorseWavelet:
    """Generalised Morse Wavelet.

    Parameters
    ----------
    beta : float
        The β parameter, controlling the bandwidth of the wavelet.
    gamma : float
        The γ parameter. The recommended setting is γ = 3 (Airy wavelet).
    norm : ['energy', 'analytic']
        The `energy` norm normalizes the wavelet to have unit L2 norm. Instead,
        the `analytic` norm makes the wavelet peak in the frequency space be
        exactly equal to 2. In the latter case the continuous wavelet transform
        value for a given scale corresponds to the analytic signal component
        (i.e. the Hilbert transform of the real signal) at the given frequency.
    """

    def __init__(self, beta=8, gamma=3, norm='analytic'):
        if beta <= 0 or gamma <= 0:
            raise ValueError('β and γ parameters must be both greater than 0.')
        if norm not in ('analytic', 'energy'):
            raise ValueError('Norm can only be `analytic` or `energy`.')

        self.beta = beta
        self.gamma = gamma
        self.norm = norm

    def psi_f(self, freqs: np.ndarray, scale: float = 1, k: int = 0):
        """Wavelet function in the frequency domain.

        By default, they are normalized to have unitary norm.

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequency values.
        scale : float
            Scale factor.
        k : int
            Order of the wavelet.
        """
        r = (2 * self.beta + 1) / self.gamma

        # I calculate the log to avoid exploding functions and improve the
        # numerical precision, particularly for the Γ(k + 1)/Γ(k + r).
        if self.norm == 'energy':
            logA = 0.5 * (np.log(np.pi * self.gamma / scale)
                          + (r + 1) * np.log(2)
                          + loggamma(k + 1)
                          - loggamma(k + r))

        else:
            logA = np.log(2 / scale) + self.beta / self.gamma * (
                1 + np.log(self.gamma / self.beta))

        # The frequency in radians
        aω = scale * 2 * np.pi * freqs

        H = np.heaviside(aω, 0.5)

        # As before, I use the log instead of doing a ω^beta.
        with np.errstate(divide='ignore', invalid='ignore'):
            wav0 = scale * H * \
                np.exp(logA - aω ** self.gamma + self.beta * np.log(aω))

        # Remove the term which explosed. Not very elegant, but it works.
        if np.isscalar(wav0):
            wav0 = 0 if np.isnan(wav0) else wav0
        else:
            wav0[np.isnan(wav0)] = 0

        # Finally, I add the generalized Laguerre polynomial term.
        return wav0 * eval_genlaguerre(k, r - 1, 2 * aω**self.gamma)

    def central_freq(self, scale=1):
        return (self.beta / self.gamma)**(1 / self.gamma) / scale * 0.5 / np.pi

    def psi_t(self, length, scale: float = 1, k: int = 0):
        """TODO

        What should we do? Find a frequency such that the wavelet psi_f goes
        to zero, then do an ifft based on that.
        """
        raise NotImplementedError('TODO')

    def cwt(self, signal: np.ndarray, scales: np.ndarray, dt=1):
        """Continuous wavelet transform.

        Parameters
        ----------
        signal : np.ndarray
            The signal to transform.
        scales : np.ndarray
            Wavelet scales.
        dt : float
            The sampling period of the signal.
        """
        # Find the fast length for the FFT
        n = len(signal)
        fast_len = fft.next_fast_len(n)

        # Signal in frequency domain
        fs = fft.fftfreq(fast_len, d=dt)
        f_sig = fft.fft(signal, n=fast_len)

        # Compute the wavelet transform
        psi = np.array([self.psi_f(fs, scale) for scale in scales])
        W = fft.ifft(f_sig * psi, n=n, workers=-1)

        freqs = self.central_freq(scales)

        return freqs, W
