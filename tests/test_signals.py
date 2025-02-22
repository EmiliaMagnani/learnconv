import numpy as np
from sampling import power_law_samples_symmetric_including_dc
from generate_input_signals import generate_frequency_localized_samples, generate_time_localized_samples_on_torus, generate_time_localized_samples
from fourier import compute_fourier_coeff



def test_fourier_squared_magnitude_equals_probability():
    """
    Test that for frequency-localized inputs X defined as
        X(t) = e^(2π i l t)
    with l drawn from a power-law with exponent=1,
    the empirical average of |(F X)(l)|^2 approximates the theoretical probability:
    
        E(|(F X)(l)|^2) ≈ p(l)
        
    where for nonzero l (taken as positive frequencies)
        p(l) = (1/l) / (2 * Σ_{j=1}^{max_value} 1/j).
    """
    # Set up parameters.
    grid_size = 2**3
    time_span = 1.0
    time_array = np.linspace(0, time_span, grid_size, endpoint=False)
    num_samples = 100000  # use many samples to get a stable average
    seed = 42
    rng = np.random.default_rng(seed)

    # Frequency-localized sample parameters.
    freq_loc_exponent = 1  # so that p(ℓ) ∝ 1/|ℓ|
    freq_max = grid_size //2
    dc_weight = 0.5  # weight for DC (we will ignore DC in this test)

    # Generate frequency-localized samples.
    X = generate_frequency_localized_samples(
        num_samples, time_array, freq_max, freq_loc_exponent,
        power_law_samples_symmetric_including_dc, rng
    )
    # X is a complex matrix of shape (len(time_array), num_samples)
    
    # Compute Fourier coefficients of each sample.
    # For a pure exponential, the FFT (with proper normalization)
    # produces a spike of magnitude 1 at the chosen frequency.
    X_fourier = compute_fourier_coeff(X, time_span)  # shape: (grid_size, num_samples)
    
    # Average squared magnitude for each frequency:
    avg_sq = np.sum(np.abs(X_fourier)**2, axis=1) / num_samples  # shape: (grid_size,)
    
    # Get frequency bins from FFT.
    freqs = np.fft.fftfreq(grid_size, d=time_span / grid_size)
    
    # We focus on nonzero frequencies.
    nonzero_mask = freqs != 0
    freqs_nonzero = np.abs(freqs[nonzero_mask])
    avg_sq_nonzero = avg_sq[nonzero_mask]
    
    # Compute theoretical probability for nonzero (positive) frequencies.
    # The candidate set for nonzero frequencies is symmetric over -freq_max,...,-1 and 1,...,freq_max.
    # For exponent = 1, the weight for frequency l is 1/|l|.
    # The total weight (over nonzero frequencies) is:
    j = np.arange(1, freq_max + 1)
    # For a positive frequency ℓ, the theoretical probability is:
    # p(ℓ) = (1/ℓ) / (dc_weight + 2 * sum_{j=1}^{freq_max} 1/j)
    total_weight = dc_weight + 2 * np.sum(1 / np.arange(1, freq_max+1))
    p_theory = (1 / freqs_nonzero) / total_weight


    # Compare the empirical average squared magnitude with the theoretical probability.
    # Allow some tolerance due to finite-sample effects.
    tol = 0.15  # 15% relative tolerance
    assert np.allclose(avg_sq_nonzero, p_theory, rtol=tol), (
        f"Empirical averaged squared magnitudes {avg_sq_nonzero} differ from theoretical probabilities {p_theory}"
    )





# Test shapes and values of generated signals

def test_generate_time_localized_samples():
    n_samples = 5
    time_array = np.linspace(0, 1, 100)
    delta = 0.1
    X = generate_time_localized_samples(n_samples, time_array, delta)

    # Expected shape: (len(time_array), n_samples)
    assert X.shape == (100, n_samples)

    # Values should be either 0 or 1/(2*delta).
    unique_vals = np.unique(X)
    expected_val = 1 / (2 * delta)
    for v in unique_vals:
        assert np.isclose(v, 0) or np.isclose(v, expected_val)

def test_generate_time_localized_samples_on_torus():
    n_samples = 5
    time_array = np.linspace(0, 1, 100)
    delta = 0.1
    X = generate_time_localized_samples_on_torus(n_samples, time_array, delta)

    # Expected shape: (len(time_array), n_samples)
    assert X.shape == (100, n_samples)

    # Values should be either 0 or 1/(2*delta).
    unique_vals = np.unique(X)
    expected_val = 1 / (2 * delta)
    for v in unique_vals:
        assert np.isclose(v, 0) or np.isclose(v, expected_val)


def test_generate_frequency_localized_samples():
    n_samples = 5
    time_array = np.linspace(0, 1, 128)
    max_value = 10
    exponent = 2.0
    rng = np.random.default_rng(42)
    # Use the symmetric version here.
    from src.sampling import power_law_samples_symmetric_including_dc

    X = generate_frequency_localized_samples(
        n_samples, time_array, max_value, exponent, power_law_samples_symmetric_including_dc, rng
    )
    # Expected shape: (len(time_array), n_samples)
    assert X.shape == (128, n_samples)
    # Check that the samples are complex.
    assert np.iscomplexobj(X)

