import numpy as np
from fourier import compute_fourier_coeff, get_fourier_coeffs
from generate_input_signals  import generate_frequency_localized_samples, generate_time_localized_samples
from sampling import power_law_samples_symmetric_including_dc

# Parameters common to both tests
GRID_SIZE = 2 ** 9
T_LEFT = 0
T_RIGHT = 1
TIME_SPAN = T_RIGHT - T_LEFT
TIME_ARRAY = np.linspace(T_LEFT, T_RIGHT, GRID_SIZE, endpoint=False)
NUM_SAMPLES = 10000  # use a large number to get a stable average
SEED = 42
rng = np.random.default_rng(SEED)

# For our tests we use a simple kernel_coeff with decay 2, so that for l != 0:
# kernel_coeff ~ |l|^{-2}. (For l==0 we set it to c0.)
# Adjust c0 and scale as needed.
KERNEL_COEFF = get_fourier_coeffs(decay=2, time_span=TIME_SPAN, n_sample_points=GRID_SIZE, c0=0.5, scale=3/(2*np.pi**2))

def _get_positive_freqs_and_sigma1(sigma_est, grid_size):
    freqs = np.fft.fftfreq(grid_size, d=TIME_SPAN/ grid_size)
    pos_mask = (freqs > 0)
    return freqs[pos_mask], sigma_est[pos_mask]

def _get_positive_freqs_and_sigma(sigma_est, grid_size, freq_max):
    freqs = np.fft.fftfreq(grid_size, d=TIME_SPAN/ grid_size)
    pos_mask = (freqs > 0) & (freqs <= freq_max)
    return freqs[pos_mask], sigma_est[pos_mask]


def test_frequency_localized_sigma_decay():
    """
    Test that for frequency-localized samples (with power-law exponent 1)
    the estimated sigma decays roughly as |l|^{-3}.
    """
    # Generate frequency-localized samples.
    # We use power_law_samples_symmetric (which yields p(l) ~ |l|^{-1})
    X = generate_frequency_localized_samples(
        NUM_SAMPLES,
        TIME_ARRAY,
        max_value=GRID_SIZE // 2,
        exponent=1,
        power_law_func=power_law_samples_symmetric_including_dc,
        rng=rng
    )
    # Compute Fourier coefficients of X.
    X_fourier = compute_fourier_coeff(X, TIME_SPAN)
    # Compute sigma_est per frequency:
    sigma_est = KERNEL_COEFF * np.sum(np.abs(X_fourier) ** 2, axis=1) / NUM_SAMPLES

    # freqs, sigma_est_pos = _get_positive_freqs_and_sigma(sigma_est, GRID_SIZE)
    freqs, sigma_est_pos = _get_positive_freqs_and_sigma(sigma_est, GRID_SIZE, freq_max=100)

    
    # We expect sigma_est_pos ~ |l|^{-3}.
    # Compute logs:
    log_freq = np.log10(freqs)
    log_sigma = np.log10(sigma_est_pos)
    
    # Fit a line: log_sigma ≈ slope * log_freq + intercept.
    slope, _ = np.polyfit(log_freq, log_sigma, 1)
    
    # Check that the slope is close to -3.
    tol = 0.5  # tolerance on the exponent
    assert abs(slope + 3) < tol, f"Expected slope ≈ -3, got {slope:.2f}"


def test_time_localized_sigma_decay():
    """
    Test that for time-localized samples with a very small δ,
    the estimated sigma decays roughly as |l|^{-2}.
    """
    # Choose a very small localization parameter δ.
    delta = 1e-3
    X = generate_time_localized_samples(NUM_SAMPLES, TIME_ARRAY, delta, seed=SEED)
    X_fourier = compute_fourier_coeff(X, TIME_SPAN)
    sigma_est = KERNEL_COEFF * np.sum(np.abs(X_fourier) ** 2, axis=1) / NUM_SAMPLES

    freqs, sigma_est_pos = _get_positive_freqs_and_sigma1(sigma_est, GRID_SIZE)
    
    # With δ very small, sinc(2πδl) ≈ 1 so we expect sigma_est_pos ~ |l|^{-2}.
    log_freq = np.log10(freqs)
    log_sigma = np.log10(sigma_est_pos)
    
    slope, _ = np.polyfit(log_freq, log_sigma, 1)
    
    tol = 0.5
    assert abs(slope + 2) < tol, f"Expected slope ≈ -2, got {slope:.2f}"
