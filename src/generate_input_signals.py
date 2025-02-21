import numpy as np


# Signal samples for input data generation


def generate_frequency_localized_samples(n_samples, time_array, max_value, exponent, power_law_func, rng):
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
        Frequency-localized sample matrix. (len(time_array),n_samples)
    """
    
    # Generate random frequencies using the power-law distribution
    frequencies = power_law_func(n_samples, max_value, exponent, rng)

    # Construct the sample matrix using cosine and sine terms for each frequency
    X = np.array([np.cos(2 * np.pi * f * time_array) + 1j * np.sin(2 * np.pi * f * time_array) 
                  for f in frequencies]).T
    
    return X


def generate_time_localized_samples(n_samples, time_array, delta, shift_center, std, rng):
    """""
    Generates a time-localized sample matrix.

    For each of the n_samples, a random time is drawn from a normal distribution
    with mean `shift_center` and standard deviation `std`. Then, for each sample,
    time points in `time_array` that lie within ±delta of the sampled time are marked
    (set to 1/(2*delta)), and the rest are set to 0.

    This returns a matrix X of shape (len(time_array), n_samples) where each column
    represents a time-localized indicator function (scaled by 1/(2*delta)).

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    time_array : np.ndarray
        1D array of time points where the samples are defined.
    delta : float
        Half-width of the localization window. A time point is considered active if
        it is within ±delta of the randomly drawn time.
    shift_center : float
        Mean of the normal distribution for sampling the random time shifts.
    std : float
        Standard deviation of the normal distribution for sampling time shifts.
    rng : np.random.Generator
        A NumPy random number generator instance.
        
    Returns
    -------
    np.ndarray
        A matrix of shape (len(time_array), n_samples) representing the time-localized samples.
    """

    # # Create a random number generator with the given seed (or use system entropy if seed is None)
    # rng = np.random.default_rng(seed)
    
    # # Generate random times using the RNG
    random_times = rng.normal(loc=shift_center, scale=std, size=n_samples)
    
    # # Create the time-localized sample matrix.
    # # For each random time 't', mark entries in time_array that are within ±delta of 't'.
    # X = np.array([
    #     np.where(((time_array - t) <= delta) & ((time_array - t) >= -delta), 1, 0)
    #     for t in random_times
    # ]).T / (2 * delta)
    
    # return X
    
    # Use broadcasting to compute the absolute difference between each time point and each random time.
    # time_array[:, None] has shape (T, 1) and random_times[None, :] has shape (1, n_samples),
    # so the result has shape (T, n_samples).
    time_diffs = np.abs(time_array[:, None] - random_times[None, :])
    
    # Create a boolean mask where True indicates the time point is within ±delta of the random time.
    mask = time_diffs <= delta
    
    # Scale the mask by 1/(2*delta) to obtain the desired amplitude.
    return mask.astype(float) / (2 * delta)



def generate_time_localized_samples_on_torus(
    n_samples: int,
    time_array: np.ndarray,
    delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates a set of time-localized indicator functions on the 1D torus.

    For each of the n_samples, we draw a uniform random shift τ in [0,1],
    then define an indicator 'bump' around τ with half-width delta. The
    distance is computed on the circle, so wrap-around is handled automatically.

    Concretely, for each point t in time_array:

        1. Compute raw distance |t - τ|.
        2. Wrap using min(|t - τ|, 1 - |t - τ|).
        3. If the wrapped distance ≤ delta, set X(t)=1/(2*delta); else 0.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate (number of 'bump' signals).
    time_array : np.ndarray
        A 1D array of time points. Usually assumed to lie within [0,1).
    delta : float
        The half-width of the localized window, in [0,1/2].
    rng : np.random.Generator
        A NumPy random generator instance for reproducible randomness.

    Returns
    -------
    X : np.ndarray
        A 2D array of shape (len(time_array), n_samples), where each column
        is one of these torus‐localized indicator bumps scaled by 1/(2*delta).
    """
    # 1) Draw random shifts τ uniformly in [0,1).
    random_shifts = rng.random(n_samples)

    # 2) For each time point t_i and each shift τ_j, compute |t_i - τ_j|.
    diffs = np.abs(time_array[:, None] - random_shifts[None, :])

    # 3) Wrap around the circle by taking min(diffs, 1 - diffs).
    circ_diffs = np.minimum(diffs, 1.0 - diffs)

    # 4) Check if circ_diffs ≤ delta => inside the 'bump'.
    mask = circ_diffs <= delta

    # 5) Scale by 1 / (2*delta) to form the final indicator bumps.
    X = mask.astype(float) / (2.0 * delta)

    return X



