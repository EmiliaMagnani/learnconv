import numpy as np
from signals import  generate_frequency_localized_samples, generate_time_localized_samples
from sampling import  power_law_samples_symmetric
from fourier import compute_fourier_coeff, compute_inverse_fourier_coeff, get_fourier_coeffs
from fourier_inference import  compute_error

seed = 42  # or any integer of choice
rng = np.random.default_rng(seed)


#number of experiments
num_experiments = 9

# Number of input functions
num_samples = 150

grid_size = 2**11 # grid points

t_left = 0
t_right = 1   # time interval

time_span = t_right - t_left

time_array = np.linspace(t_left,t_right,grid_size)
noise = .01 # noise level in the data
const = 1e-1 # constant for the regularization term lambda

# Ground truth function (target) and observation  noise         

#parameter for target signal
target_decay = 2.51  # decay rate of the target signal  ~ 1/target_decay

target_coeff= get_fourier_coeffs(target_decay, time_span, grid_size, c0=1, scale=1)
target = np.fft.ifft(target_coeff*grid_size).real

# choose the kernel
kernel_decay =  2 # decay rate of the  complex exponential kernel   #1.01
kernel_coeff = get_fourier_coeffs(kernel_decay, time_span, grid_size, c0=1/2, scale=3/(2*np.pi**2))


# freq-loc input signals parameters
freq_loc_inputs_decay = 1 
freq_max = grid_size // 2

r_freq_loc = 1 / 3
b_freq_loc = 3

# freq-loc inputs
X_freq_loc = generate_frequency_localized_samples(num_samples, time_array, freq_max, freq_loc_inputs_decay, power_law_samples_symmetric, seed=seed)

#compute error for freq-loc input signals
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
    const,
    generate_frequency_localized_samples,
    sample_gen_params_freq_loc,
)

#time-loc input signals parameters 
loc_parameter = 0.001
r_time_loc = 1 / 2
b_time_loc = 2

#time-loc inputs
X_time_loc = generate_time_localized_samples(num_samples, time_array, loc_parameter,seed=seed+1)

#compute error for time-loc input signals
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
    const,
    generate_time_localized_samples,
    sample_gen_params_time_loc,
)

# Create parameter strings for each type of signal
freq_loc_params = f"n{num_samples}_noise{noise}_targetdecay{target_decay}_kerdecay{kernel_decay}_inputdecay{freq_loc_inputs_decay}_b{b_freq_loc}_r{r_freq_loc}"

time_loc_params = f"n{num_samples}_noise{noise}_targetdecay{target_decay}_kerdecay{kernel_decay}_locparam{loc_parameter}_b{b_time_loc}_r{r_time_loc}"

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

