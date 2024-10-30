import numpy as np

def compute_fourier_coeff(signal_values,time_span):
    """
    Computes the Fourier coefficients of a given signal.

    Parameters:
    ----------
    signal_values : numpy.ndarray
        The sampled values of the signal. It can be a 1D array (for a single signal) 
        or a 2D array (for multiple signals along different columns).
    time_span : float
        The total length of the time interval over which the signal is sampled. 
        Typically, this is the difference between the start and end points of the time domain.

    Returns:
    -------
    numpy.ndarray
        The Fourier coefficients of the input signal, normalized by the length 
        of the signal and scaled by the time interval.
    
    Notes:
    -----
    This function assumes that the input signal is sampled uniformly over the time interval.
    The Fourier transform is computed using NumPy's Fast Fourier Transform (FFT) algorithm.
    """
    num_samples = len(signal_values)
    return np.fft.fft(signal_values, axis=0) * time_span / num_samples

