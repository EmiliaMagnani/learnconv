from typing import Literal
import numpy as np
from sampling import power_law_samples_symmetric_including_dc
from fourier import compute_fourier_coeff
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


def test_frequency_decay_rate():
    """
    Test that the decay of E[|F{X}(l)|^2]for frequency-localized inputs
    follows a power law decay. since we use a power law with exponent -1 for the
    frequency distribution, we expect the Fourier coefficients to decay as 1/|l|.
    
    In other words, on a log-log scale, we expect:
      log(E[|F{X}(l)|^2]) ~ -log(|l|) + constant,
    meaning the slope should be close to -1.
    """
    import numpy as np
    
    # Set up parameters.
    grid_size = 2**9
    time_span = 1.0
    time_array = np.linspace(0, time_span, grid_size, endpoint=False)
    num_samples = 10000  # Many samples for a stable average.
    seed = 42
    rng = np.random.default_rng(seed)
    
    # Frequency-localized sample parameters.
    freq_loc_exponent = 1  # exponent α = 1 implies p(ℓ) ~ 1/|ℓ|
    freq_max = grid_size

    # Generate frequency-localized samples.
    X = generate_frequency_localized_samples(
        num_samples, time_array, freq_max, freq_loc_exponent,
        power_law_samples_symmetric_including_dc, rng
    )
    # X has shape (len(time_array), num_samples)
    
    # Compute Fourier coefficients.
    X_fourier = compute_fourier_coeff(X, time_span)  # shape: (grid_size, num_samples)
    
    # Compute average squared magnitude at each Fourier frequency.
    avg_sq = np.sum(np.abs(X_fourier)**2, axis=1) / num_samples  # shape: (grid_size,)
    
    # Get Fourier frequency bins.
    freqs = np.fft.fftfreq(grid_size, d=(time_span / grid_size))
    
    # Focus on nonzero frequencies.
    nonzero_mask = np.abs(freqs) > 0
    effective_freqs = np.abs(freqs[nonzero_mask])
    avg_sq_nonzero = avg_sq[nonzero_mask]
    
    # Log-transform the effective frequencies and the corresponding energy.
    log_freqs = np.log(effective_freqs)
    log_energy = np.log(avg_sq_nonzero)
    
    # Fit a line to the log-log data.
    slope, intercept = np.polyfit(log_freqs, log_energy, 1)
    
    tol = 0.1  # Tolerance for the slope.
    assert abs(slope + 1) < tol, (
        f"Decay slope is not -1 as expected: found slope = {slope:.3f} (tolerance ±{tol})"
    )

   