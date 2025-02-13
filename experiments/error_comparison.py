import numpy as np
from signals import  generate_frequency_localized_samples, generate_time_localized_samples, construct_sine_series_signal
from sampling import  power_law_samples_symmetric
from fourier import compute_fourier_coeff, compute_inverse_fourier_coeff, get_fourier_coeffs, get_fourier_coeffs_balanced
from regularization import compute_lambda
from fourier_inference import  compute_prediction, compute_error


seed = 42  # or any integer of choice
rng = np.random.default_rng(seed)


# Number of input functions
num_samples = 1000
num_experiments = 9 # number of experiments for each sample to compute error bars

grid_size = 2**8 # grid points

t_left = 0
t_right = 1  # time interval

time_span = t_right - t_left

time_array = np.linspace(t_left,t_right,grid_size,endpoint=False)
noise = .55 # noise level in the data


# Compute frequency bins 
freqs = np.fft.fftfreq(grid_size, time_span / grid_size)

balancing_vector = np.ones(grid_size)
balancing_vector[freqs != 0] = np.abs(freqs[freqs != 0]) ** 1
balancing_vector[1] =.3
balancing_vector[-1] =.3
balancing_vector[2] = .5
balancing_vector[-2] = .5
balancing_vector[3] = 3
balancing_vector[-3] = 3
balancing_vector[4] = 4
balancing_vector[-4] = 4
balancing_vector[5] = 4
balancing_vector[-5] = 4
balancing_vector[6] = 2
balancing_vector[-6] = 2
target_coeff = get_fourier_coeffs_balanced(decay=3.51, time_span=time_span, n_sample_points=grid_size, c0=.5, scale=.7, balancing_vector=balancing_vector)
# target_coeff= get_fourier_coeffs(target_decay, time_span, grid_size, c0=1, scale=1)

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

# Now compute the target function via the inverse Fourier transform.
target = np.fft.ifft(target_coeff * grid_size).real

# choose the kernel
kernel_decay =  2 # decay rate of the  complex exponential kernel   
kernel_coeff = get_fourier_coeffs(kernel_decay, time_span, grid_size, c0=1/2, scale=3/(2*np.pi**2))


# freq-loc input signals 
freq_loc_inputs_decay = 1 
freq_max = grid_size // 2

r_freq_loc = 1 / 3
b_freq_loc = 3
X_freq_loc = generate_frequency_localized_samples(num_samples, time_array, freq_max, freq_loc_inputs_decay, power_law_samples_symmetric, seed=seed)
const_lam = 1e-4


#time-loc input signals 
loc_parameter = 0.001
r_time_loc = 1 / 2
b_time_loc = 2
X_time_loc = generate_time_localized_samples(num_samples, time_array, loc_parameter,seed=seed+1)

# # error computation for frequency localised signals

sample_gen_params_freq_loc = {
    "max_value": freq_max,
    "exponent": freq_loc_inputs_decay,
    "power_law_func": power_law_samples_symmetric,
    "seed": rng,   ## to get error bars you need rng.. not seed!
}

error_squared_sampmean_freq_loc, error_squared_sampstd_freq_loc = compute_error(
    num_samples,
    num_experiments,
    time_array,
    time_span,
    kernel_coeff,
    target_coeff,
    noise,
    r_freq_loc,
    b_freq_loc,
    const_lam,
    generate_frequency_localized_samples,
    sample_gen_params_freq_loc,
    optimize_lambda=True  
)



## error computation for time localised signals

sample_gen_params_time_loc = {
    "delta": loc_parameter,
    "seed": rng,
}

error_squared_sampmean_time_loc, error_squared_sampstd_time_loc = compute_error(
    num_samples,
    num_experiments,
    time_array,
    time_span,
    kernel_coeff,
    target_coeff,
    noise,
    r_time_loc,
    b_time_loc,
    const_lam,
    generate_time_localized_samples,
    sample_gen_params_time_loc,
    optimize_lambda=True  
)

# Create parameter strings for each type of signal
freq_loc_params = f"n{num_samples}_noise{noise}_seed{seed}_grid_size{grid_size}_const_lam{const_lam}_inputdecay{freq_loc_inputs_decay}_b{b_freq_loc}_r{r_freq_loc}"

time_loc_params = f"n{num_samples}_noise{noise}_seed{seed}_gird_size{grid_size}_const_lam{const_lam}_locparam{loc_parameter}_b{b_time_loc}_r{r_time_loc}"

# Save frequency-localized results with freq-loc specific parameters
np.save(f'results/freq_loc_error_squared_sampmean_{freq_loc_params}.npy', error_squared_sampmean_freq_loc)
np.save(f'results/freq_loc_error_squared_sampstd_{freq_loc_params}.npy', error_squared_sampstd_freq_loc)
# np.save(f'results/freq_loc_error_squared_logmean_{freq_loc_params}.npy', error_squared_logmean_freq_loc)
# np.save(f'results/freq_loc_error_squared_logstd_{freq_loc_params}.npy', error_squared_logstd_freq_loc)

# Save time-localized results with time-loc specific parameters
np.save(f'results/time_loc_error_squared_sampmean_{time_loc_params}.npy', error_squared_sampmean_time_loc)
np.save(f'results/time_loc_error_squared_sampstd_{time_loc_params}.npy', error_squared_sampstd_time_loc)
# np.save(f'results/time_loc_error_squared_logmean_{time_loc_params}.npy', error_squared_logmean_time_loc)
# np.save(f'results/time_loc_error_squared_logstd_{time_loc_params}.npy', error_squared_logstd_time_loc)

