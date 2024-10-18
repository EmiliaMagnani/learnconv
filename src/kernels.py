import numpy as np


def complex_exponential_kernel(input_values, decay_rate, num_terms):
    """
    Computes the kernel function using complex exponentials, which includes
    both positive and negative frequency components.

    Parameters:
    ----------
    input_values : numpy.ndarray or float
        Input time or position values where the kernel function is evaluated.
    decay_rate : float
        Exponent controlling the decay of the kernel function.
    num_terms : int
        Number of terms to sum in the series. This controls the truncation of the series.

    Returns:
    -------
    numpy.ndarray or float
        The computed kernel function at each value of `input_values`.
    """
    term_indices_pos = np.arange(1, num_terms + 1).reshape(-1, 1)
    term_indices_neg = np.arange(-num_terms, 0).reshape(-1, 1)

    # Calculate the positive and negative frequency terms
    positive_terms = np.sum(np.exp(1j * 2 * term_indices_pos * np.pi * input_values) / (term_indices_pos ** decay_rate), axis=0)
    negative_terms = np.sum(np.exp(1j * 2 * term_indices_neg * np.pi * input_values) / (np.abs(term_indices_neg) ** decay_rate), axis=0)

    return 1 + positive_terms + negative_terms



# def complex_exponential_kernel(t,gamma,M):
    # return 1 + sum((np.exp(1j*2*k*np.pi*t) / (k**gamma)) for k in range(1,M+1)) + sum((np.exp(1j*2*k*np.pi*t) / (np.abs(k)**gamma)) for k in range(-M,0))



def cosine_kernel(input_values, decay_rate, num_terms):
    """
    Computes the kernel function using cosine terms, which includes only 
    positive frequency components.

    Parameters:
    ----------
    input_values : numpy.ndarray or float
        Input time or position values where the kernel function is evaluated.
    decay_rate : float
        Exponent controlling the decay of the cosine kernel function.
    num_terms : int
        Number of terms to sum in the series. This controls the truncation of the series.

    Returns:
    -------
    numpy.ndarray or float
        The computed cosine kernel function at each value of `input_values`.
    """
    term_indices = np.arange(1, num_terms + 1).reshape(-1, 1)
    cosine_terms = np.sum(2 * np.cos(2 * term_indices * np.pi * input_values) / (term_indices ** decay_rate), axis=0)
    
    return 1 + cosine_terms


# def kernel_cosines(t,gamma,M):
#     return 1 + sum((2* np.cos(2*k*np.pi*t) / (k**gamma)) for k in range(1,M+1)) 