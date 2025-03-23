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

def power_law_samples_symmetric_including_dc(n_samples, max_value, exponent, rng, dc_weight=.5):
    """
    Generate samples from a symmetric power-law distribution over candidate frequencies,
    including the DC (zero-frequency) component.
    
    The candidate frequencies are constructed as:
        [0, -max_value, -max_value+1, ..., -1, 1, ..., max_value]
    For each candidate frequency x, a weight is assigned as follows:
        - For x == 0 (the DC component), the weight is set to dc_weight.
        - For x != 0, the weight is computed as 1 / |x|**exponent.
    
    These weights are normalized to form a probability distribution over the candidate set.
    Then, n_samples frequencies are drawn randomly according to this probability distribution.
    
    Parameters
    ----------
    n_samples : int
        The number of frequency samples to generate.
    max_value : int
        The maximum absolute frequency value. The candidate set consists of integers from -max_value
        to max_value, with 0 added separately.
    exponent : float
        The exponent in the power-law decay. Larger values yield lower probabilities for higher frequencies.
    rng : numpy.random.Generator
        A random number generator instance (e.g., created with np.random.default_rng(seed)).
        This is used to ensure reproducible sampling.
    dc_weight : float, optional
        The weight assigned to the zero frequency (DC component). Default is 0.5.
    
    Returns
    -------
    samples : numpy.ndarray
        A one-dimensional array of length n_samples containing the sampled frequencies.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> samples = power_law_samples_symmetric_including_dc(5, 10, 2.0, rng, dc_weight=0.5)
    >>> samples
    array([  0,  -1,   2,  -3,   1])
    """
    import numpy as np
    
    # Create candidate frequencies, now including 0.
    candidates = np.concatenate(([0], np.arange(-max_value, 0), np.arange(1, max_value + 1)))
    
    # Compute weights for each candidate.
    weights = []
    for x in candidates:
        if x == 0:
            weights.append(dc_weight)
        else:
            weights.append(1.0 / (abs(x) ** exponent))
    weights = np.array(weights)
    
    # Normalize to form a probability distribution.
    probabilities = weights / np.sum(weights)
    
    # Draw samples using the computed probabilities.
    samples = rng.choice(candidates, size=n_samples, p=probabilities)
    return samples



def power_law_samples_symmetric(n_samples, max_value, exponent, rng):
    """
    Generates a specified number of samples drawn from a symmetric power-law probability distribution 
    on the integers [-max_value, -1] ∪ [1, max_value] with a given exponent.
    
    The probability decays as:
    
    p(x) proportional to 1/{|x|^{exponent}}
    
    
    for $x \in \{-{max_value}, ... -1, 1, ..., \{max_value}\}$.
    
    Parameters:
    ----------
    n_samples : int
        The number of samples to generate.
    max_value : int
        The maximum absolute value in the range from which samples are drawn (inclusive).
    exponent : float
        The exponent of the power-law distribution. Higher values lead to more weight on smaller numbers.
    rng : numpy.random.Generator
        Random number generator instance (from `np.random.default_rng()`).

    Returns:
    -------
    numpy.ndarray
        An array of `n_samples` integers drawn from the symmetric power-law distribution.
    """
    import numpy as np
    
    # Create candidate values: negative values from -max_value to -1 and positive values from 1 to max_value.
    candidates = np.concatenate((np.arange(-max_value, 0), np.arange(1, max_value + 1)))
    
    # Define the power-law probability function using absolute value.
    def power_law_prob(x):
        return 1.0 / (abs(x) ** exponent)
    
    # Compute the normalization constant.
    normalization_constant = sum(power_law_prob(x) for x in candidates)
    
    # Compute probabilities for each candidate.
    probabilities = np.array([power_law_prob(x) / normalization_constant for x in candidates])
    
    # Draw samples from the candidates using the computed probabilities.
    samples = rng.choice(candidates, size=n_samples, p=probabilities)
    
    return samples

