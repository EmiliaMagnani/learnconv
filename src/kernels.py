import numpy as np


# List of kernels in time space 

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





# define the kernel k(t) = 3t^2 -3t +1 
def periodic_sobolev(t):
    """
    Computes the polynomial kernel function k(t) = 3t^2 - 3t + 1 for a given input value or array of input values.
    
    Parameters:
    ----------
    t : float or numpy.ndarray
        The input values where the kernel is evaluated.
    
    Returns:
    -------
    numpy.ndarray or float
        The computed periodic Sobolev kernel function at each input value.
    """
    return 3 * t**2 - 3 * t + 1



def cosine_kernel(t, decay_rate, num_terms):
    """
    Computes the kernel function using cosine terms, which includes only 
    positive frequency components.

    Parameters:
    ----------
    t : numpy.ndarray or float
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
    cosine_terms = np.sum(2* np.cos(2 * term_indices * np.pi * t) / (term_indices ** decay_rate), axis=0)
    
    return 1 + cosine_terms


# def kernel_cosines(t,gamma,M):
#     return 1 + sum((2* np.cos(2*k*np.pi*t) / (k**gamma)) for k in range(1,M+1)) 

import numpy as np

def dirichlet_kernel(t, order, L=1):
    """
    Computes the Dirichlet kernel on the torus for a given order ,
    periodic with period 1 and defined in [-L, L].
    
    Parameters:
    ----------
    t : numpy.ndarray or float
        The angle(s) on the torus (typically in radians).
    order : int
        The order of the Dirichlet kernel (number of harmonics).
    L : float, optional
        The length of the interval [-L, L] where the function is defined. Default is 1.
    
    Returns:
    -------
    numpy.ndarray or float
        The value of the Dirichlet kernel at each angle theta, periodic with period 1.
    """
    # Map theta to the interval [-L, L]
    t = np.mod(t + L, 2 * L) - L
    
    # Handle the case when theta is very close to zero (to avoid division by zero)
    epsilon = 1e-12
    t = np.where(np.abs(t) < epsilon, epsilon, t)
    
    # Compute the Dirichlet kernel
    kernel_values = np.sin((order + 0.5) * t * np.pi / L) / np.sin(0.5 * t * np.pi / L)
    
    return kernel_values

import numpy as np

def dirichlet_kernel_shifted(t, order, L):
    """
    Computes the Dirichlet kernel on the torus for a given order, 
    periodic with period 2π and defined in [0, 2L] with the maximum at L.
    
    Parameters:
    ----------
    t : numpy.ndarray or float
        The angle(s) on the torus (typically in radians).
    order : int
        The order of the Dirichlet kernel (number of harmonics).
    L : float, optional
        The half-length of the interval [0, 2L] where the function is defined. Default is 1.
    
    Returns:
    -------
    numpy.ndarray or float
        The value of the Dirichlet kernel at each angle, periodic with period 2π and maximum at L.
    """
    # Map input_values to the interval [0, 2L]
    t = np.mod(t, 2 * L)
    
    # Shift input values to place the maximum at L
    shifted_input = t - L

    # Handle cases where shifted_input is very close to zero (to avoid division by zero)
    epsilon = 1e-12
    shifted_input = np.where(np.abs(shifted_input) < epsilon, epsilon, shifted_input)
    
    # Compute the Dirichlet kernel with the maximum at L
    kernel_values = np.sin((order + 0.5) * np.pi * shifted_input / L) / np.sin(0.5 * np.pi * shifted_input / L)
    
    return kernel_values



def periodic_kernel(t, p, sigma=1.0, length_scale=1.0):
    """
    Computes the periodic kernel for an input value or array of input values.
    
    Parameters:
    ----------
    t : float or numpy.ndarray
        The input values where the kernel is evaluated.
    p : float
        The period of the kernel.
    sigma : float, optional
        The variance (amplitude) of the kernel. Default is 1.0.
    length_scale : float, optional
        The length scale of the kernel, controlling smoothness. Default is 1.0.
    
    Returns:
    -------
    numpy.ndarray or float
        The computed periodic kernel for the input values.
    """
    # Compute the sine-based distance to capture periodicity
    dist = np.sin(np.pi * t / p)
    
    # Compute the periodic kernel
    return sigma**2 * np.exp(-2 * (dist**2) / length_scale**2)


def exp_decay_torus(input_values,gamma):
    return gamma / (gamma + (np.sin(np.pi*input_values))**2)
