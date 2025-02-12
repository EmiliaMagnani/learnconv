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


def compute_inverse_fourier_coeff(F, time_span):
    """
    Reconstructs the time-domain signal from Fourier coefficients computed by compute_fourier_coeff.
    
    Parameters:
    -----------
    F : numpy.ndarray
        Fourier coefficients of the signal, as computed by compute_fourier_coeff.
    time_span : float
        The total time span (T) over which the signal is defined.
    
    Returns:
    --------
    numpy.ndarray
        The reconstructed time-domain signal.
        
    Notes:
    ------
    The function assumes that F was computed as:
    
        F = np.fft.fft(signal_values, axis=0) * time_span / num_samples
    
    Hence, the inverse transform scales the result of np.fft.ifft by (num_samples/time_span)
    to recover the original signal.
    """
    num_samples = len(F)
    # Compute the inverse FFT and multiply by (N/T) to undo the forward scaling.
    reconstructed_signal = np.fft.ifft(F, axis=0) * (num_samples / time_span)
    return reconstructed_signal


def get_fourier_coeffs(decay, time_span, n_sample_points, c0, scale):
    """
    Constructs a vector of Fourier coefficients for a signal (or kernel), where:
      - The zero-frequency (DC) coefficient is set to c0.
      - For nonzero frequencies, the coefficients decay as 1/|l|^decay,
        scaled by the factor 'scale'.
    
    Parameters:
    -----------
    decay : float
        The exponent that controls the decay rate of the Fourier coefficients.
    time_span : float
        The total time span (period) of the signal.
    n_sample_points : int
        The number of sample points (and Fourier coefficients).
    c0 : float, optional
        The Fourier coefficient at l = 0. (Default is 1/2.)
    scale : float, optional
        The scaling factor for the nonzero Fourier coefficients.
        
    Returns:
    --------
    F : numpy.ndarray (complex)
        The array of Fourier coefficients, arranged in the order used by np.fft.ifft.
    """
    # Compute the sampling interval.
    d = time_span / n_sample_points
    
    # Get the frequency bins: these tell us which Fourier mode corresponds to each index.
    freqs = np.fft.fftfreq(n_sample_points, d=d)
    
    # Allocate the Fourier coefficient vector.
    F = np.empty(n_sample_points, dtype=complex)
    
    # Set coefficients: for l = 0, use c0; for l != 0, use scale/|l|^decay.
    # for k, l in enumerate(freqs):
    #     if l == 0:
    #         F[k] = c0
    #     else:
    #         F[k] = scale / (abs(l)**decay)

    # vectorized: 
    F[freqs == 0] = c0
    nonzero = freqs != 0
    F[nonzero] = scale / (np.abs(freqs[nonzero]) ** decay)

    return F


def get_fourier_coeffs_balanced(decay, time_span, n_sample_points, c0, scale, balancing_vector=None):
    """
    Constructs a vector of Fourier coefficients for a signal. For the nonzero
    frequencies the coefficients are computed as:
    
        F[l] = balancing_vector[l] * (scale / |l|^decay)
    
    where the balancing factor is an explicit balancing_vector (which must have
    length n_sample_points). When n_sample_points is even, the Nyquist frequency
    (at index n_sample_points//2) is forced to be real to ensure a real time-domain
    signal.
    
    Parameters
    ----------
    decay : float
        The exponent controlling the base decay.
    time_span : float
        The total time span (period) of the signal.
    n_sample_points : int
        The number of sample points (and Fourier coefficients).
    c0 : float
        The Fourier coefficient at zero frequency.
    scale : float
        The base scaling factor.
    balancing_vector : array-like, optional
        A vector of balancing factors for each frequency.
        
    Returns
    -------
    F : numpy.ndarray (complex)
        The array of Fourier coefficients arranged in the order used by np.fft.ifft.
    """
    # Compute the sampling interval.
    d = time_span / n_sample_points
    # Get the frequency bins.
    freqs = np.fft.fftfreq(n_sample_points, d=d)
    
    # Allocate the Fourier coefficient vector.
    F = np.empty(n_sample_points, dtype=complex)
    
    # Set the zero-frequency (DC) term.
    F[freqs == 0] = c0
    
    # Identify nonzero frequency indices.
    nonzero = freqs != 0
    
    # Base coefficients (without the balancing factor)
    base_coeffs = scale / (np.abs(freqs[nonzero]) ** decay)
    
    # Apply balancing: if a balancing_vector is provided, use it; otherwise, use the base coefficients.
    if balancing_vector is not None:
        balancing_vector = np.asarray(balancing_vector)
        if balancing_vector.shape[0] != n_sample_points:
            raise ValueError("balancing_vector must have length equal to n_sample_points")
        F[nonzero] = balancing_vector[nonzero] * base_coeffs
    else:
        F[nonzero] = base_coeffs

    # For an even number of sample points, ensure the Nyquist frequency is real.
    if n_sample_points % 2 == 0:
        nyquist_index = n_sample_points // 2
        F[nyquist_index] = F[nyquist_index].real

    return F




