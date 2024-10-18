import numpy as np

def power_law_samples(n_samples, max_value, exponent, rng):
    """
    Generates a specified number of samples drawn from a power-law probability distribution 
    on the integers [1, max_value] with a given exponent.
    
    Parameters:
    ----------
    n_samples : int
        The number of samples to generate.
    max_value : int
        The maximum value in the range from which samples are drawn (inclusive).
    exponent : float
        The exponent of the power-law distribution. Higher values lead to more 
        weight on smaller numbers.
    rng : numpy.random.Generator
        Random number generator instance (from `np.random.default_rng()`).

    Returns:
    -------
    numpy.ndarray
        An array of `n_samples` integers drawn from the power-law distribution.
    """
    def power_law_prob(x):
        return 1.0 / (x ** exponent)
    
    normalization_constant = sum(power_law_prob(l) for l in range(1, max_value + 1))
    probabilities = np.array([power_law_prob(l) / normalization_constant for l in range(1, max_value + 1)])
    
    samples = rng.choice(np.arange(1, max_value + 1), size=n_samples, p=probabilities)
    return samples

