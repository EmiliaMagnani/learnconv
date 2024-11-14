import numpy as np 

from fourier import compute_fourier_coeff
from signal_functions import  generate_frequency_localized_samples, generate_time_localized_samples
from sampling import power_law_samples

def compute_output_data_matrix(X_fourier, target_f_coeff, noise_level, n_grid_points, time_interval):
    """
    Computes the output data matrix Y using the Fourier coefficients of the input matrix.
    
    Parameters:
    ----------
    X_fourier : numpy.ndarray
        Fourier coefficients of the input matrix.
    target_f_coeff : numpy.ndarray
        True Fourier coefficients of the target function.
    noise_level : float
        Standard deviation of the noise added to the output.
    n_grid_points : int
        Number of grid points (resolution).
    
    Returns:
    -------
    numpy.ndarray
        The computed output data matrix Y with noise added.
    """
    # Compute the inverse FFT for all samples at once
    Y = np.fft.ifft(target_f_coeff[:, np.newaxis] * X_fourier, axis=0) * (n_grid_points / time_interval)
    
    # Add noise to the entire matrix at once
    noise = noise_level * np.random.normal(0, 1, Y.shape)
    Y += noise
    
    return Y


def run_inference_time_localized(num_samples, time_array, time_span, kernel_coeff, target_fourier_coeff, noise, alpha, b, loc_parameter):
    """
    Runs inference for time-localized signals and returns the prediction in the time domain.

    Parameters:
    ----------
    num_samples : int
        Number of input samples for inference.
    time_array : numpy.ndarray
        Array of time points.
    time_span : float
        Total time span.
    kernel_coeff : numpy.ndarray
        Kernel coefficients used in the eigenvalue problem.
    target_fourier_coeff : numpy.ndarray
        True Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise added to the output.
    alpha : float
        Regularization exponent.
    b : float
        Source conditions parameter.
    loc_parameter : float
        Time localization parameter for generating time-localized samples.

    Returns:
    -------
    numpy.ndarray
        The prediction in the time domain (inverse Fourier of the predicted coefficients).
    """
    # Generate time-localized samples
    X = generate_time_localized_samples(num_samples, time_array, loc_parameter)

    # Compute Fourier coefficients of X and Y
    X_fourier = compute_fourier_coeff(X, time_span)
    Y = compute_output_data_matrix(X_fourier, target_fourier_coeff, noise, time_array.size, time_span)
    Y_fourier = compute_fourier_coeff(Y, time_span)

    # Regularization and Fourier prediction
    lamb =  num_samples ** (-1 / (2 * alpha + 1 + 1/b))   #1e-4 *
    prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)

    for l in range(time_array.size):
        eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / num_samples
        term1 = kernel_coeff[l] / (eigenval + lamb)
        term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / num_samples
        prediction_fourier[l] =  (term1 * term2)

    # Compute inverse FFT to get prediction in the time domain
    prediction_time_domain = np.fft.ifft(prediction_fourier) * (time_array.size / time_span)
    
    return prediction_fourier, np.real(prediction_time_domain)

def run_inference_frequency_localized(num_samples, time_array, time_span, kernel_coeff, target_fourier_coeff, noise, alpha, b, freq_loc_inputs_decay, freq_max, seed=None):
    """
    Runs inference for frequency-localized signals and returns the prediction in the time domain.

    Parameters:
    ----------
    num_samples : int
        Number of samples for inference.
    time_array : numpy.ndarray
        Array of time points.
    time_span : float
        Total time span.
    kernel_coeff : numpy.ndarray
        Kernel coefficients used in the eigenvalue problem.
    target_fourier_coeff : numpy.ndarray
        True Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise added to the output.
    alpha : float
        Regularization exponent.
    b : float
        Source conditions parameter.
    freq_loc_inputs_decay : float
        Exponent for the power-law distribution.
    freq_max : int
        Maximum frequency value for generating frequency-localized samples.
    seed : int, optional
        Seed for random number generation (default is None).

    Returns:
    -------
    numpy.ndarray, numpy.ndarray
        the Fourier coefficients of the prediction and the The prediction in the time domain (inverse Fourier of the predicted coefficients).
    """
    # Generate frequency-localized samples
    X = generate_frequency_localized_samples(num_samples, time_array, freq_max, freq_loc_inputs_decay, power_law_samples, seed=seed)

    # Compute Fourier coefficients of X and Y
    X_fourier = compute_fourier_coeff(X, time_span)
    Y = compute_output_data_matrix(X_fourier, target_fourier_coeff, noise, time_array.size, time_span)
    Y_fourier = compute_fourier_coeff(Y, time_span)

    # Regularization and Fourier prediction
    lamb = num_samples ** (-1 / (2 * alpha + 1 + 1/b))   #1e-4 *
    prediction_fourier_coeff = np.zeros(time_array.size, dtype=np.complex128)

    for l in range(time_array.size):
        eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / num_samples
        term1 = kernel_coeff[l] / (eigenval + lamb)
        term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / num_samples
        prediction_fourier_coeff[l] = 2*(term1 * term2)   # 2 factor probably because you cover only half of the frequencies (positive frequencies)
    

    # Compute inverse FFT to get prediction in the time domain
    prediction_time_domain = np.fft.ifft(prediction_fourier_coeff) * (time_array.size / time_span)   
    fft_pred_fourier_coeff = compute_fourier_coeff(np.real(prediction_time_domain), time_span)

    return fft_pred_fourier_coeff, prediction_time_domain


def run_inference_error_time_localized(num_samples, num_experiments, time_array, time_span, kernel_coeff, target_fourier_coeff, noise, alpha, b, series_truncation, loc_parameter):
    """
    Runs the inference and error computation for time-localized signals.

    Parameters:
    ----------
    num_samples : int
        Number of samples for inference.
    num_experiments : inti
        Number of experiments to average over.
    time_array : numpy.ndarray
        Array of time points.
    time_span : float
        Total time span.
    kernel_coeff : numpy.ndarray
        Kernel coefficients used in the eigenvalue problem.
    target_fourier_coeff : numpy.ndarray
        True Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise added to the output.
    alpha : float
        Regularization exponent.
    b: float
        Source conditions parameter.
    series_truncation : int
        Number of terms in the (kernel) series truncation for error computation.
    loc_parameter : float
        Time localization parameter for generating time-localized samples.

    Returns:
    -------
    error_sampmean : numpy.ndarray
        Mean error for each sample.
    error_sampstd : numpy.ndarray
        Standard deviation of error for each sample.
    error_logmean : numpy.ndarray
        Log mean of the error for each sample.
    error_logstd : numpy.ndarray
        Log standard deviation of error for each sample.
    """
    error_squared_sampmean = np.zeros(num_samples)
    error_squared_sampstd = np.zeros(num_samples)
    error_squared_logmean = np.zeros(num_samples)
    error_squared_logstd = np.zeros(num_samples)
    
    for n in range(1, num_samples + 1):
        error_of_experiments = np.zeros(num_experiments)
        
        for j in range(num_experiments):
            # Generate time-localized samples
            X = generate_time_localized_samples(n, time_array, loc_parameter)

            # Compute Fourier coefficients of X and Y
            X_fourier = compute_fourier_coeff(X, time_span)
            Y = compute_output_data_matrix(X_fourier, target_fourier_coeff, noise, time_array.size, time_span)
            Y_fourier = compute_fourier_coeff(Y, time_span)

            # Regularization and error computation
            lamb = num_samples ** (-1 / (2 * alpha + 1 + 1/b))   #1e-4 *
            prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)
            
            for l in range(time_array.size):
                eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / n
                term1 = kernel_coeff[l] / (eigenval + lamb)
                term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / n
                prediction_fourier[l] = term1 * term2

            # Compute the H error
            w_diff_coeff = np.abs(prediction_fourier) - np.abs(target_fourier_coeff)
            error_h_squared = (np.abs(w_diff_coeff[:series_truncation]) ** 2 / kernel_coeff[:series_truncation]).sum()
            error_of_experiments[j] = error_h_squared
        
        # Error statistics
        error_squared_sampmean[n - 1] = np.mean(error_of_experiments)
        error_squared_sampstd[n - 1] = np.std(error_of_experiments)
        error_squared_logmean[n - 1] = np.mean(np.log(error_of_experiments))
        error_squared_logstd[n - 1] = np.std(np.log(error_of_experiments))
    
    return  error_squared_sampmean, error_squared_sampstd, error_squared_logmean, error_squared_logstd



def run_inference_error_frequency_localized(num_samples, num_experiments, time_array, time_span, kernel_coeff, target_fourier_coeff, noise,  alpha, b, series_truncation, freq_loc_inputs_decay, freq_loc_max, seed=None):
    """
    Runs the inference and error computation for frequency-localized signals.

    Parameters:
    ----------
    num_samples : int
        Number of samples for inference.
    num_experiments : int
        Number of experiments to average over.
    time_array : numpy.ndarray
        Array of time points.
    time_span : float
        Total time span.
    kernel_coeff : numpy.ndarray
        Kernel coefficients used in the eigenvalue problem.
    target_fourier_coeff : numpy.ndarray
        True Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise added to the output.
    alpha : float
        Regularization exponent.
    b : float
        Source conditions parameter.
    series_truncation : int
        Number of terms in the (kernel) series truncation for error computation.
    freq_loc_inputs_decay : float
        Exponent for the power-law distribution (used for frequency-localized signals).
    freq_loc_max : int
        Maximum frequency value for generating frequency-localized samples.
    seed : int, optional
        Seed for random number generation (default is None).

    Returns:
    -------
    error_sampmean : numpy.ndarray
        Mean error for each sample.
    error_sampstd : numpy.ndarray
        Standard deviation of error for each sample.
    error_logmean : numpy.ndarray
        Log mean of the error for each sample.
    error_logstd : numpy.ndarray
        Log standard deviation of error for each sample.
    """
    error_squared_sampmean = np.zeros(num_samples)
    error_squared_sampstd = np.zeros(num_samples)
    error_squared_logmean = np.zeros(num_samples)
    error_squared_logstd = np.zeros(num_samples)
    
    for n in range(1, num_samples + 1):
        error_squared_exp = np.zeros(num_experiments)
        
        for j in range(num_experiments):
            # Generate frequency-localized samples
            X = generate_frequency_localized_samples(n, time_array, freq_loc_max, freq_loc_inputs_decay, power_law_samples, seed=seed)

            # Compute Fourier coefficients of X and Y
            X_fourier = compute_fourier_coeff(X, time_span)
            Y = compute_output_data_matrix(X_fourier, target_fourier_coeff, noise, time_array.size, time_span)
            Y_fourier = compute_fourier_coeff(Y, time_span)

            # Regularization and error computation
            lamb = num_samples ** (-1 / (2 * alpha + 1 + 1/b))   # 1e-4*
            prediction_fourier_coeff = np.zeros(time_array.size, dtype=np.complex128)
            
            for l in range(time_array.size):
                eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / num_samples
                term1 = kernel_coeff[l] / (eigenval + lamb)
                term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / num_samples
                prediction_fourier_coeff[l] = 2*(term1 * term2)   # 2 factor probably because you cover only half of the frequencies (positive frequencies)

            prediction_time_domain = np.fft.ifft(prediction_fourier_coeff) * (time_array.size / time_span)   
            fft_pred_fourier_coeff = compute_fourier_coeff(np.real(prediction_time_domain), time_span)

            # Compute the H error
            w_diff_coeff = np.abs(fft_pred_fourier_coeff) - np.abs(target_fourier_coeff)
            error_h_squared = (np.abs(w_diff_coeff[:series_truncation]) ** 2 / kernel_coeff[:series_truncation]).sum()
            error_squared_exp[j] = error_h_squared
        
        # Error statistics
        error_squared_sampmean[n - 1] = np.mean(error_squared_exp)
        error_squared_sampstd[n - 1] = np.std(error_squared_exp)
        error_squared_logmean[n - 1] = np.mean(np.log(error_squared_exp))
        error_squared_logstd[n - 1] = np.std(np.log(error_squared_exp))
    
    return  error_squared_sampmean, error_squared_sampstd, error_squared_logmean, error_squared_logstd











################################################################################################

# def compute_output_data_matrix_old(X_fourier, target_f_coeff, noise_level, n_grid_points, n_samples):
#     """
#     Computes the output data matrix Y using the Fourier coefficients of the input matrix.
    
#     Parameters:
#     ----------
#     X_fourier : numpy.ndarray
#         Fourier coefficients of the input matrix.
#     target_f_coeff : numpy.ndarray
#         True Fourier coefficients of the target function.
#     noise_level : float
#         Standard deviation of the noise added to the output.
#     n_grid_points : int
#         Number of grid points (resolution).
    
#     Returns:
#     -------
#     numpy.ndarray:  (n_grid_points, n_samples)
#         The computed output data matrix Y with noise added.
#     """

#     Y = np.zeros((n_grid_points, n_samples))  # Initialize the output data matrix

#     for i in range(n_samples):
#         Y[:, i] = np.fft.ifft(n_grid_points * target_f_coeff * X_fourier[:, i])  + noise_level * np.random.normal(0, 1, n_grid_points)
    
#     return Y


