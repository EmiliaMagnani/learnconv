import numpy as np
from generate_input_signals import (
    generate_frequency_localized_samples,
    generate_time_localized_samples_on_torus,
)
from sampling import power_law_samples_symmetric_including_dc
from fourier import (
    get_fourier_coeffs,
    get_fourier_coeffs_balanced,
    compute_fourier_coeff,
)
from fourier_inference import compute_prediction


seed = 42  # or any integer of choice
rng = np.random.default_rng(seed)

# Number of input functions
num_samples = 50

grid_size = 2**9  # grid points

t_left = 0
t_right = 1  # time interval

time_span = t_right - t_left

time_array = np.linspace(t_left, t_right, grid_size, endpoint=False)
noise = 0.45  # noise level in the data


# Compute frequency bins
freqs = np.fft.fftfreq(grid_size, time_span / grid_size)

balancing_vector = np.ones(grid_size)
balancing_vector[freqs != 0] = np.abs(freqs[freqs != 0]) ** 1
# set the first frequencies of the balancing vector as you like
balancing_vector[1] = balancing_vector[-1] = 0.3
balancing_vector[2] = balancing_vector[-2] = 0.5
balancing_vector[3] = balancing_vector[-3] = 3
balancing_vector[4] = balancing_vector[-4] = 4
balancing_vector[5] = balancing_vector[-5] = 4
balancing_vector[6] = balancing_vector[-6] = 2
target_coeff = get_fourier_coeffs_balanced(
    decay=3.51,
    time_span=time_span,
    n_sample_points=grid_size,
    c0=0.5,
    scale=0.7,
    balancing_vector=balancing_vector,
)

# Ensure frequency 0 is real.
target_coeff[0] = target_coeff[0].real

# Identify indices for positive frequencies.
pos_indices = np.where(freqs > 0)[0]

# Generate random phases for these frequencies.
random_phases = np.exp(1j * 2 * np.pi * rng.random(len(pos_indices)))
target_coeff[pos_indices] *= random_phases

# Enforce conjugate symmetry for negative frequencies.
neg_indices = np.where(freqs < 0)[0]
# The corresponding negative frequency for a positive frequency at index i is at index -i.
target_coeff[neg_indices] = np.conjugate(target_coeff[-neg_indices])

# (If grid_size is even, the Nyquist frequency must also be real.)
if grid_size % 2 == 0:
    nyquist_index = grid_size // 2
    target_coeff[nyquist_index] = target_coeff[nyquist_index].real


# choose the kernel
# we construct the kernel coefficients with decay rate 2
kernel_decay = 2
kernel_coeff = get_fourier_coeffs(
    kernel_decay, time_span, grid_size, c0=0.5, scale=3 / (2 * np.pi**2)
)


# freq-loc input signals
freq_loc_inputs_decay = 1
freq_max = grid_size
r_freq_loc = 1 / 3
b_freq_loc = 3
const_lam = 1e-4
X_freq_loc = generate_frequency_localized_samples(
    num_samples,
    time_array,
    freq_max,
    freq_loc_inputs_decay,
    power_law_samples_symmetric_including_dc,
    rng,
)
X_freq_loc_fourier = compute_fourier_coeff(X_freq_loc, time_span)


# time-loc input signals
loc_parameter = 0.002
r_time_loc = 1 / 2
b_time_loc = 2
X_time_loc = generate_time_localized_samples_on_torus(
    num_samples, time_array, loc_parameter, rng
)
X_time_loc_fourier = compute_fourier_coeff(X_time_loc, time_span)


## prediciton for frequency localised signals
prediction_four_coeff_freq_loc, prediction_freq_loc = compute_prediction(
    num_samples=num_samples,
    time_array=time_array,
    time_span=time_span,
    kernel_coeff=kernel_coeff,
    target_coeff=target_coeff,
    noise=noise,
    lamb=None,
    X_fourier=X_freq_loc_fourier,
    optimize_lambda=True,
)

## prediction for time localised signals
prediction_four_coeff_time_loc, prediction_time_loc = compute_prediction(
    num_samples=num_samples,
    time_array=time_array,
    time_span=time_span,
    kernel_coeff=kernel_coeff,
    target_coeff=target_coeff,
    noise=noise,
    lamb=None,
    X_fourier=X_time_loc_fourier,
    optimize_lambda=True,
)

# Create parameter strings for each type of signal
freq_loc_params = (
    f"n{num_samples}_grid_size{grid_size}_seed{seed}_noise{noise}_freqmax{freq_max}"
)

time_loc_params = f"n{num_samples}_grid_size{grid_size}_seed{seed}_noise{noise}_locparam{loc_parameter}"

#save the results 
# np.save(f"learnconv_results/freq_loc_pred_{freq_loc_params}.npy", prediction_freq_loc)
# np.save(f"learnconv_results/time_loc_pred_{time_loc_params}.npy", prediction_time_loc)