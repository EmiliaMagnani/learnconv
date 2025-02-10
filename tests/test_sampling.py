from src.sampling import power_law_samples,power_law_samples_symmetric
from src.signals import generate_time_localized_samples,generate_frequency_localized_samples
import numpy as np


def test_power_law_samples():
    n_samples = 1000
    max_value = 10
    exponent = 2.0
    rng = np.random.default_rng(42)
    samples = power_law_samples(n_samples, max_value, exponent, rng)

    # All samples should be between 1 and max_value.
    assert samples.min() >= 1
    assert samples.max() <= max_value


def test_power_law_samples_symmetric():
    n_samples = 1000
    max_value = 10
    exponent = 2.0
    rng = np.random.default_rng(42)
    samples = power_law_samples_symmetric(n_samples, max_value, exponent, rng)

    # Samples should lie in the set {-max_value,...,-1,1,...,max_value}.
    assert samples.min() >= -max_value
    assert samples.max() <= max_value
    assert 0 not in samples


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


def test_generate_frequency_localized_samples():
    n_samples = 5
    time_array = np.linspace(0, 1, 128)
    max_value = 10
    exponent = 2.0
    # Use the symmetric version here.
    from src.sampling import power_law_samples_symmetric

    X = generate_frequency_localized_samples(
        n_samples, time_array, max_value, exponent, power_law_samples_symmetric, seed=42
    )
    # Expected shape: (len(time_array), n_samples)
    assert X.shape == (128, n_samples)
    # Check that the samples are complex.
    assert np.iscomplexobj(X)
