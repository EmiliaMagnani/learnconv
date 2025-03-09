from typing import Literal
import numpy as np
from sampling import power_law_samples_symmetric_including_dc
from fourier import compute_fourier_coeff, get_fourier_coeffs
from generate_input_signals import generate_frequency_localized_samples, generate_time_localized_samples_on_torus
import pytest


@pytest.mark.parametrize("delta, grid_size", [
    (0.001, 2**12),  (0.002, 2**9) # Example: very small delta, grid_size=1024
])
def test_small_delta_bumps_fourier_is_close_to_one(delta: float, grid_size:int):
    """
    For small delta, the low-frequency Fourier coefficients of the
    time-localized bumps should have magnitude ~ 1. In theory,
    |X_hat(ell)| = sinc(2πδell) ~ 1 for |ell| << 1/delta.

    Here, we only check that those are close to 1, ignoring the exact sinc shape.
    """
    seed = 421
    rng = np.random.default_rng(seed)

    # Time on [0,1)
    t_left, t_right = 0.0, 1.0
    time_span = t_right - t_left
    time_array = np.linspace(t_left, t_right, grid_size, endpoint=False)

    # Generate signals
    num_samples = 150  # or however many you like
    X_time_loc = generate_time_localized_samples_on_torus(
        n_samples=num_samples,
        time_array=time_array,
        delta=delta,
        rng=rng
    )
    # Shape = (grid_size, num_samples)

    # Compute Fourier coeffs
    X_time_loc_fourier = compute_fourier_coeff(X_time_loc, time_span)
    # Shape = (grid_size, num_samples) along axis=0

    # Frequencies
    freqs = np.fft.fftfreq(grid_size, d=(time_span / grid_size))  
    # e.g. for grid_size=1024: freq indices [0,...,511, -512,..., -1]

    # We'll pick a small band of frequencies around 0, for which
    # we expect the magnitude to be close to 1. For instance, take
    # ell_max = 5, 10, or something like 1/(10*delta) if that is smaller
    ell_max = min(grid_size // 2, int(.5 / (2 * np.pi * delta))) # in fact  ∣ℓ∣≪1/δ
    print('ellmax=', ell_max)
    freq_mask = (freqs >= -ell_max) & (freqs <= ell_max)

    # Extract those Fourier coefficients
    F_sub = X_time_loc_fourier[freq_mask, :]  # shape (#freqs_in_range, num_samples)
    F_mags = np.abs(F_sub)**2

    # One way: check the average magnitude across samples
    avg_mags_per_freq = F_mags.mean(axis=1)

    # We want to assert they are "close to 1" for these frequencies.
    # The tolerance can be adjusted depending on how precise we expect them to be.
    close_to_one = np.isclose(avg_mags_per_freq, 1.0, atol=0.05, rtol=0.05)
    # For instance, we allow ±0.05 absolute difference or 10% relative difference.

    # If you want to ensure *all* these frequencies are close to 1, do:
    assert np.all(close_to_one), (
        "Low-frequency magnitude of time-localized bumps is not close to 1 "
        f"for delta={delta} and freq range |ell|<={ell_max}."
    )

    # Alternatively, if you want to check the entire 2D array individually,
    # you can do something like:
    # all_close = np.isclose(F_mags, 1.0, atol=0.05, rtol=0.1)
    # assert np.all(all_close), "All low-frequency coefficients should be near 1."







# def test_empirical_energy_around_one():
#     """
#     Generate many bump signals with random shifts on the torus.
#     Compute the average squared amplitude of their Fourier transform
#     at ALL frequencies, i.e. sum(|X_fourier|^2)/n. 
#     Check that these are all close to 1.0.
#     """
#     rng = np.random.default_rng(123)
    
#     # 1) Parameters
#     grid_size = 2**10
#     time_span = 1.0
#     time_array = np.linspace(0, 1, grid_size, endpoint=False)
#     n_samples = 100
#     delta = 0.001
    
#     # 2) Generate signals
#     X = generate_time_localized_samples_on_torus(n_samples, time_array, delta, rng)
#     # shape: (grid_size, n_samples)

#     # 3) Fourier transform of X
#     X_fourier = compute_fourier_coeff(X, time_span)  # shape (grid_size, n_samples)

#     # 4) Average squared amplitude for each freq (0..grid_size-1).
#     #    This yields a (grid_size,) array.
#     avg_sq = np.sum(np.abs(X_fourier) ** 2, axis=1) / n_samples   #shape (grid_size,)
#     abs_error = np.abs(avg_sq - 1.0)
#     print(avg_sq)


#     assert np.all(abs_error < .1), "Not close to 1"





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
