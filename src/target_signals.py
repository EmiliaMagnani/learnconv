import numpy as np


# Ground truth signals

def construct_even_fourier_signal(t, decay_rate, num_terms):
    """
    Constructs an even function f(t) as a truncated cosine Fourier series.
    
    The function is defined by:
        f(t) = 1/2 + 2 * sum_{ell=1}^{num_terms} (1 / ell^decay_rate) * cos(2*pi*ell*t)
    
    Parameters:
    -----------
    t : numpy.ndarray
        1D array of time values on which to evaluate the signal.
    decay_rate : float
        The exponent controlling the decay of the Fourier coefficients for ell>=1.
    num_terms : int
        Number of cosine terms to include (the sum runs from ell = 1 to ell = num_terms).
    
    Returns:
    --------
    signal : numpy.ndarray
        The constructed signal evaluated at each point in t.
    """
    # Start with the DC term f(t) = 1/2
    signal = np.full_like(t, 1/2, dtype=float)
    
    # Sum over cosine terms with the specified decay
    for ell in range(1, num_terms + 1):
        signal += 2 * (1 / ell**decay_rate) * np.cos(2 * np.pi * ell * t)
    
    return signal

def construct_sine_series_signal(t, decay_rate, num_terms):
    """
    Computes a truncated sine series approximation of a function using the sum of scaled sine terms.

    Parameters:
    ----------
    t : numpy.ndarray or float
        The points (e.g., time or position) where the sine series is evaluated.
    decay_rate : float
        The exponent that controls the decay rate of the sine terms.
    num_terms : int
        The number of sine terms to include in the summation.

    Returns:
    -------
    numpy.ndarray or float
        The computed sine series approximation at each value of `input_points`.
    """
    return sum((2 * np.sin(2 * j * np.pi * t) / (j ** decay_rate)) for j in range(1, num_terms + 2))



def truncated_fourier_series(input_points, decay_rate, num_terms):
    """
    Computes a truncated Fourier series approximation with both positive and negative frequency components.

    Parameters:
    ----------
    input_points : numpy.ndarray or float
        The points (e.g., time or position) where the Fourier series is evaluated.
    decay_rate : float
        The exponent controlling the decay rate of the terms.
    num_terms : int
        The number of terms to include in each of the positive and negative frequency summations.

    Returns:
    -------
    numpy.ndarray or float
        The computed Fourier series approximation at each value of `input_points`.
    """
    # Compute the positive frequency terms and convert to an array
    positive_terms = sum(np.exp(1j * 2 * k * np.pi * input_points) / (k ** decay_rate) for k in range(1, num_terms + 2))
    
    # Compute the negative frequency terms and convert to an array
    negative_terms = sum(np.exp(1j * 2 * k * np.pi * input_points) / (k ** decay_rate) for k in range(-num_terms, 0))
    
    # Return the sum of positive and negative terms
    return positive_terms + negative_terms



def piecewise_linear_signal(input_points, period):
    """
    Generates a periodic, piecewise linear signal with a specified period.

    Parameters:
    ----------
    input_points : numpy.ndarray or float
        The input values (can be a single float or a NumPy array) where the signal is evaluated.
    period : float, optional
        The period of the piecewise linear signal. 
    Returns:
    -------
    numpy.ndarray or float
        The generated piecewise linear signal at the specified input points.
    """
    # Normalize input points by the period to create periodic behavior
    normalized_points = input_points / period

    # Generate the piecewise linear signal using modulo operation
    linear_signal = 2 * (normalized_points - np.floor(0.5 + normalized_points))
    
    return linear_signal


def compute_heat_kernel(x, t):
    """
    Compute the heat kernel (Green's function) for the heat equation.

    The heat kernel is given by:
        G(x, t) = exp(-x^2 / (4t)) / sqrt(4πt)
    
    Parameters:
    -----------
    x : float or np.ndarray
        The spatial coordinate(s).
    t : float
        The time variable (must be positive).
    
    Returns:
    --------
    float or np.ndarray
        The value of the heat kernel evaluated at x and t.
    """
    t = np.asarray(t)  # Ensure t is an array
    if np.any(t <= 0):
        raise ValueError("Time 't' must be positive.")
    
    return np.exp(-x**2 / (4 * t)) / np.sqrt(4 * np.pi * t)


