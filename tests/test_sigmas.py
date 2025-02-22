import numpy as np
from sampling import power_law_samples_symmetric_including_dc
from fourier import compute_fourier_coeff, get_fourier_coeffs
from generate_input_signals import generate_frequency_localized_samples, generate_time_localized_samples_on_torus
import pytest

import numpy as np
import pytest
from generate_input_signals import generate_time_localized_samples_on_torus
from fourier import compute_fourier_coeff

def test_empirical_energy_around_one():
    """
    Generate many bump signals with random shifts on the torus.
    Compute the average squared amplitude of their Fourier transform
    at ALL frequencies, i.e. sum(|X_fourier|^2)/n. 
    Check that these are all close to 1.0.
    """
    rng = np.random.default_rng(123)
    
    # 1) Parameters
    grid_size = 2**10
    time_span = 1.0
    time_array = np.linspace(0, 1, grid_size, endpoint=False)
    n_samples = 100
    delta = 0.001
    
    # 2) Generate signals
    X = generate_time_localized_samples_on_torus(n_samples, time_array, delta, rng)
    # shape: (grid_size, n_samples)

    # 3) Fourier transform of X
    X_fourier = compute_fourier_coeff(X, time_span)  # shape (grid_size, n_samples)

    # 4) Average squared amplitude for each freq (0..grid_size-1).
    #    This yields a (grid_size,) array.
    avg_sq = np.sum(np.abs(X_fourier) ** 2, axis=1) / n_samples   #shape (grid_size,)
    abs_error = np.abs(avg_sq - 1.0)
    print(avg_sq)


    assert np.all(abs_error < .1), "Not close to 1"


























# def test_frequency_eigenvalues():
#     """
#     Test that for frequency-localized inputs the empirical eigenvalues satisfy
#        sigma_l = K_hat(l) * E[|F{X}(l)|^2]  ≈  K_hat(l) * p_l,
#     where the theoretical probability for nonzero frequencies (with exponent=1)
#        p(l) = 1/|l| / (2 * sum_{j=1}^{max_value} 1/j).
#     """
#     # Set up parameters.
#     grid_size = 2**9
#     time_span = 1.0
#     time_array = np.linspace(0, time_span, grid_size, endpoint=False)
#     num_samples = 10000 # Use many samples to get a stable average.
#     seed = 42
#     rng = np.random.default_rng(seed)

#     # Frequency-localized sample parameters.
#     freq_loc_exponent = 1  # exponent α = 1 → p(l) ~ 1/|l|
#     freq_max = 100
#     dc_weight = 0.5  # weight for DC (this value influences p(0) but we will test nonzero frequencies)

#     # Kernel parameters: choose a kernel with decay = 2, scale = 1, so that for l != 0, 
#     #  K_hat(l) = 1/|l|^2.
#     kernel_decay = 2
#     c0 = 0.5
#     scale = 3/(2*np.pi**2)
#     kernel_coeff = get_fourier_coeffs(kernel_decay, time_span, grid_size, c0, scale)

#     # Generate frequency-localized samples.
#     X = generate_frequency_localized_samples(
#         num_samples, time_array, freq_max, freq_loc_exponent,
#         power_law_samples_symmetric_including_dc, rng
#     )
#     # X is of shape (len(time_array), num_samples)
    
#     # Compute Fourier coefficients of each sample.
#     # Our compute_fourier_coeff is assumed to be defined so that for a pure exponential,
#     # the FFT yields a spike of magnitude 1 at the corresponding frequency.
#     X_fourier = compute_fourier_coeff(X, time_span)  # shape: (grid_size, num_samples)
    
#     # For a pure exponential sample, |F{X}(l)|^2 is 1 at the chosen frequency and 0 elsewhere.
#     # Thus, the average squared magnitude at each frequency approximates the probability that
#     # that frequency was chosen.
#     avg_sq = np.sum(np.abs(X_fourier) ** 2, axis=1) / num_samples  # shape: (grid_size,)

#     # Now, the empirical eigenvalues are:
#     sigma_emp = kernel_coeff * avg_sq  # elementwise multiplication

#     # Get the frequency bins from FFT.
#     freqs = np.fft.fftfreq(grid_size, d=time_span / grid_size)
#     # We will test only positive frequencies (excluding DC).
#     # pos_mask = freqs > 0
#     # freqs_pos = freqs[pos_mask]
#     pos_mask = (freqs > 0) & (freqs <= freq_max)
#     freqs_pos = freqs[pos_mask]
#     avg_sq_pos = avg_sq[pos_mask]
#     kernel_pos = kernel_coeff[pos_mask]
#     sigma_emp_pos = sigma_emp[pos_mask]
    
#     # Compute the theoretical probability for positive frequencies.
#     # In our candidate set (constructed in power_law_samples_symmetric_including_dc), the nonzero candidates are:
#     #   [-freq_max, -freq_max+1, ..., -1, 1, 2, ..., freq_max].
#     # The weight for nonzero x is 1/|x|^exponent. For exponent=1, that's 1/|x|.
#     # The total weight for nonzero frequencies is:
#     j = np.arange(1, freq_max + 1)
#     total_nonzero_weight = .5 + 2 * np.sum(1 / j)  # factor 2 for positive and negative.
#     # Thus, for a positive frequency l, the theoretical probability is:
#     p_theory = (1 / freqs_pos) / total_nonzero_weight

#     # Now, check that the empirical average squared magnitude approximates the theoretical probability.
#     tol = 0.15  # 10% relative tolerance.
#     for i, l in enumerate(freqs_pos):
#         emp_prob = avg_sq_pos[i]
#         theo_prob = p_theory[i]
#         assert np.isclose(emp_prob, theo_prob, rtol=tol), (
#             f"For frequency {l:.1f}, empirical probability {emp_prob:.3e} "
#             f"differs from theoretical {theo_prob:.3e}"
#         )
    
#     # Finally, the theoretical eigenvalue should be:
#     sigma_theory = kernel_pos * p_theory
#     for i, l in enumerate(freqs_pos):
#         emp_sigma = sigma_emp_pos[i]
#         theo_sigma = sigma_theory[i]
#         assert np.isclose(emp_sigma, theo_sigma, rtol=tol), (
#             f"For frequency {l:.1f}, empirical eigenvalue {emp_sigma:.3e} "
#             f"differs from theoretical {theo_sigma:.3e}"
#         )
