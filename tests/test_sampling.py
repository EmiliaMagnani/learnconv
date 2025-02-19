from sampling import power_law_samples, power_law_samples_symmetric,power_law_samples_symmetric_including_dc
from generate_input_signals import generate_time_localized_samples,generate_frequency_localized_samples
import numpy as np
from scipy.stats import chisquare


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

def test_power_law_samples_decay():
    """
    Test that the nonzero samples from power_law_samples_symmetric_including_dc
    follow a power-law decay proportional to l^{-1}.

    Specifically, when exponent=1, the probability for a nonzero frequency l should be:
       p(l) ~ 1/|l|.
    We generate a large number of samples, compute the empirical probabilities,
    and then fit a line (in log–log scale) to confirm the slope is approximately -1.
    """
    # Parameters for the test.
    n_samples = 10000
    max_value = 170
    exponent = 1
    dc_weight = 0.5  # default value used in the function
    rng = np.random.default_rng(1234)

    # Generate samples. The candidates include 0.
    samples = power_law_samples_symmetric_including_dc(n_samples, max_value, exponent, rng, dc_weight=dc_weight)
    
    # Exclude the DC (zero) value, because we want to check the decay for nonzero frequencies.
    nonzero_samples = samples[samples != 0]
    
    # Take absolute values to group negative and positive frequencies together.
    abs_samples = np.abs(nonzero_samples)
    
    # Count occurrences for each frequency value l = 1, 2, ..., max_value.
    # np.bincount returns an array where index i corresponds to the count of i.
    counts = np.bincount(abs_samples)
    # counts[0] corresponds to 0 but we ignore it; the relevant counts are for l>=1.
    l_values = np.arange(1, len(counts))
    empirical_prob = counts[1:] / np.sum(counts[1:])  # normalize to get probabilities

    # Fit a line to the log-log plot: log10(empirical_prob) vs log10(l_values)
    log_l = np.log10(l_values)
    log_prob = np.log10(empirical_prob)
    slope, intercept = np.polyfit(log_l, log_prob, 1)
    
    tol = 0.2  # tolerance on the slope estimate
    assert abs(slope + 1) < tol, f"Expected slope ≈ -1, got {slope:.2f}"


def test_power_law_samples_decay_chisquare():
    """
    Test that the nonzero samples from power_law_samples_symmetric_including_dc
    follow a power-law decay proportional to 1/|l| by comparing the observed counts
    to the expected counts (via a chi-square test).
    """
    # Use a larger number of samples and a smaller candidate range for robustness.
    n_samples = 100000  
    max_value = n_samples//2     # reduce range so that bins are well-populated
    exponent = 1
    dc_weight = 0.5
    rng = np.random.default_rng(1234)

    # Generate samples.
    samples = power_law_samples_symmetric_including_dc(n_samples, max_value, exponent, rng, dc_weight=dc_weight)
    
    # Exclude the DC component.
    nonzero_samples = samples[samples != 0]
    abs_samples = np.abs(nonzero_samples)
    
    # Count occurrences for l = 1, 2, ..., max_value.
    counts = np.bincount(abs_samples, minlength=max_value+1)[1:]  # ignore index 0
    
    # Theoretical probabilities for l=1,...,max_value (only nonzero frequencies)
    l_values = np.arange(1, max_value + 1)
    theoretical_prob = 1 / l_values
    theoretical_prob /= theoretical_prob.sum()  # normalize
    
    # Compute the expected counts based on nonzero sample count.
    nonzero_count = nonzero_samples.size
    expected_counts = nonzero_count * theoretical_prob
    
    # Perform chi-square goodness-of-fit test.
    chi2, p_value = chisquare(counts, f_exp=expected_counts)
    
    # Assert that the p-value is high (fail if the observed counts deviate too much).
    assert p_value > 0.05, f"Chi-square test failed: chi2={chi2:.2f}, p-value={p_value:.3f}"


