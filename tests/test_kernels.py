from kernels import periodic_sobolev
import numpy as np

def test_periodic_sobolev_scalar():
    t = 0.5
    # k(t) = 3*t^2 - 3*t + 1; for t=0.5 this should be 0.25.
    result = periodic_sobolev(t)
    expected = 3 * (0.5 ** 2) - 3 * 0.5 + 1
    np.testing.assert_allclose(result, expected)


def test_periodic_sobolev_array():
    t = np.array([0, 0.5, 1])
    expected = 3 * t**2 - 3 * t + 1
    result = periodic_sobolev(t)
    np.testing.assert_allclose(result, expected)

