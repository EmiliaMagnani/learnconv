import numpy as np


# Signal samples and data generation

def generate_frequency_localized_samples(n_samples, time_array, max_value, exponent, power_law_func, seed=None):
    """
    Generates frequency-localized samples using random frequencies drawn from a power-law distribution.

    Parameters:
    ----------
    n_samples : int
        Number of samples to generate.
    time_array : numpy.ndarray
        Array of time points where the signals are evaluated.
    max_frequency : int
        Maximum frequency value to sample.
    exponent : float
        Exponent for the power-law distribution.
    power_law_func : callable
        Function to generate power-law samples. It should accept arguments 
        for the number of samples, max frequency, and exponent.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        
    Returns:
    -------
    numpy.ndarray
        Frequency-localized sample matrix.
    """
    # Create a local random generator with the provided seed
    rng = np.random.default_rng(seed)
    
    # Generate random frequencies using the power-law distribution
    frequencies = power_law_func(n_samples, max_value, exponent, rng)

    # Construct the sample matrix using cosine and sine terms for each frequency
    X = np.array([np.cos(2 * np.pi * f * time_array) + 1j * np.sin(2 * np.pi * f * time_array) 
                  for f in frequencies]).T
    
    return X



def generate_time_localized_samples(n_samples, time_array, delta):
    """
    Generates time-localized samples using a normal distribution and creates a time-localized array.
    
    Parameters:
    ----------
    n_samples : int
        Number of samples to generate.
    time_array : numpy.ndarray
        Array of time points where the samples will be localized.
    delta : float
        The localization factor.

    Returns:
    -------
    numpy.ndarray
        Time-localized sample matrix.
    """
    random_times = np.random.normal(0.5, 0.16, n_samples)
    X = np.array([np.where(((time_array - t) <= 2 * delta) & ((time_array - t) >= 0), 1, 0) 
                  for t in random_times]).T / (2 * delta)
    return X



# Ground truth signals

def truncated_sine_series(input_points, decay_rate, num_terms):
    """
    Computes a truncated sine series approximation of a function using the sum of scaled sine terms.

    Parameters:
    ----------
    input_points : numpy.ndarray or float
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
    return sum((2 * np.sin(2 * j * np.pi * input_points) / (j ** decay_rate)) for j in range(1, num_terms + 2))


def truncated_fourier_series(input_points, decay_rate, num_terms):
    """
    Computes a truncated Fourier series approximation using exponential terms.

    Parameters:
    ----------
    input_points : numpy.ndarray or float
        The points (e.g., time or position) where the Fourier series is evaluated.
    decay_rate : float
        The exponent that controls the decay rate of the Fourier coefficients.
    num_terms : int
        The number of terms to include in the summation.

    Returns:
    -------
    numpy.ndarray or float
        The computed Fourier series approximation at each value of `input_points`,
        using both positive and negative frequency components.
    """
    positive_terms = (sum(np.exp(1j * 2 * k * np.pi * input_points) / (k ** decay_rate)) for k in range(1, num_terms + 2))
    negative_terms = (sum(np.exp(1j * 2 * k * np.pi * input_points) / (k ** decay_rate)) for k in range(-num_terms, 0))
    
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
