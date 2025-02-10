import numpy as np
from src.fourier import compute_fourier_coeff, compute_inverse_fourier_coeff, get_fourier_coeffs

def test_fourier_inverse():
    # Create a simple sine signal.
    t = np.linspace(0, 1, 128, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t)
    time_span = 1.0

    # Compute Fourier coefficients and then reconstruct.
    F = compute_fourier_coeff(signal, time_span)
    reconstructed_signal = compute_inverse_fourier_coeff(F, time_span).real

    np.testing.assert_allclose(signal, reconstructed_signal, rtol=1e-5, atol=1e-5)

def test_get_fourier_coeffs():
    time_span = 1.0
    n_points = 64
    c0 = 2.0
    scale = 1.0
    decay = 2.0

    F = get_fourier_coeffs(decay, time_span, n_points, c0, scale)
    d = time_span / n_points
    freqs = np.fft.fftfreq(n_points, d=d)

    # Check that the DC component equals c0.
    idx_dc = np.where(freqs == 0)[0]
    np.testing.assert_allclose(F[idx_dc], c0)

    # For nonzero frequencies, check the decay.
    for idx, freq in enumerate(freqs):
        if freq != 0:
            expected = scale / (abs(freq) ** decay)
            np.testing.assert_allclose(F[idx], expected)
