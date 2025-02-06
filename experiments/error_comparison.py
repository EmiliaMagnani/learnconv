import numpy as np
from fourier import compute_fourier_coeff
from signal_functions import truncated_sine_series
from fourier import compute_fourier_coeff
from kernels import complex_exponential_kernel
from inference_utils import  run_inference_error_frequency_localized, run_inference_error_time_localized


#number of experiments
num_experiments = 9

#CHOOSE number of input functions
num_samples = 10

grid_size = 2**12 # grid points

t_left = 0
t_right = 1

time_span = t_right - t_left

time_array = np.linspace(t_left,t_right,grid_size)

#parameter for target signal
target_decay_rate = 2.51 # decay rate of the target signal  
noise = .01 # noise level

# Ground truth function and observation  noise         
sum_terms_target = 100  # index  of truncation of the fourier series
target = truncated_sine_series(
    input_points=time_array, decay_rate=target_decay_rate, num_terms=sum_terms_target
)

##fourier coefficients of target
target_fourier_coeff = compute_fourier_coeff(
    target, time_span
)  # normalized by the number of grid points


# complex exponential kernel
kernel_decay_rate =  2 # decay rate of the  complex exponential kernel   #1.01
sum_terms_kernel = sum_terms_target
evaluated_kernel = complex_exponential_kernel(time_array, kernel_decay_rate, sum_terms_kernel)
kernel_coeff = compute_fourier_coeff(evaluated_kernel, time_span)


# freq-loc input signals parameters
freq_loc_inputs_decay = 1
freq_max = sum_terms_target

alpha_freq_loc = 1/3
b_freq_loc = 3

#compute error for freq-loc input signals
error_squared_sampmean_freq_loc, error_squared_sampstd_freq_loc, error_squared_logmean_freq_loc, error_squared_logstd_freq_loc = run_inference_error_frequency_localized(
    num_samples=num_samples,
    num_experiments= num_experiments,
    time_array=time_array,
    time_span=time_span,
    kernel_coeff= kernel_coeff,
    target_fourier_coeff=target_fourier_coeff,
    noise=noise,
    alpha=alpha_freq_loc,
    b=b_freq_loc,
    series_truncation=sum_terms_target,
    freq_loc_inputs_decay=freq_loc_inputs_decay,
    freq_loc_max=freq_max
)

#time-loc input signals parameters 
loc_parameter = 0.0001
alpha_time_loc = .5
b_time_loc = 2

#compute error for time-loc input signals
error_squared_sampmean_time_loc, error_squared_sampstd_time_loc, error_squared_logmean_time_loc, error_squared_logstd_time_loc = run_inference_error_time_localized(
    num_samples=num_samples,
    num_experiments= num_experiments,
    time_array=time_array,
    time_span=time_span,
    kernel_coeff= kernel_coeff,
    target_fourier_coeff=target_fourier_coeff,
    noise=noise,
    alpha=alpha_time_loc,
    b=b_time_loc,
    series_truncation=sum_terms_target,
    loc_parameter=loc_parameter
)

# Create parameter strings for each type of signal
freq_loc_params = f"n{num_samples}_noise{noise}_targetdecay{target_decay_rate}_kerdecay{kernel_decay_rate}_alpha{alpha_freq_loc}_b{b_freq_loc}_inputdecay{freq_loc_inputs_decay}"

time_loc_params = f"n{num_samples}_noise{noise}_targetdecay{target_decay_rate}_kerdecay{kernel_decay_rate}_alpha{alpha_time_loc}_b{b_time_loc}_locparam{loc_parameter}"

# Save frequency-localized results with freq-loc specific parameters
np.save(f'results/freq_loc_error_squared_sampmean_{freq_loc_params}.npy', error_squared_sampmean_freq_loc)
np.save(f'results/freq_loc_error_squared_sampstd_{freq_loc_params}.npy', error_squared_sampstd_freq_loc)
np.save(f'results/freq_loc_error_squared_logmean_{freq_loc_params}.npy', error_squared_logmean_freq_loc)
np.save(f'results/freq_loc_error_squared_logstd_{freq_loc_params}.npy', error_squared_logstd_freq_loc)

# Save time-localized results with time-loc specific parameters
np.save(f'results/time_loc_error_squared_sampmean_{time_loc_params}.npy', error_squared_sampmean_time_loc)
np.save(f'results/time_loc_error_squared_sampstd_{time_loc_params}.npy', error_squared_sampstd_time_loc)
np.save(f'results/time_loc_error_squared_logmean_{time_loc_params}.npy', error_squared_logmean_time_loc)
np.save(f'results/time_loc_error_squared_logstd_{time_loc_params}.npy', error_squared_logstd_time_loc)

