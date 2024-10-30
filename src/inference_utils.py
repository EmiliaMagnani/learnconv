import numpy as np 

from fourier import compute_fourier_coeff
from signal_functions import  generate_frequency_localized_samples, generate_time_localized_samples
from sampling import power_law_samples

def compute_output_data_matrix(X_fourier, target_f_coeff, noise_level, n_grid_points):
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
    Y = np.fft.ifft(n_grid_points * target_f_coeff[:, np.newaxis] * X_fourier, axis=0)
    
    # Add noise to the entire matrix at once
    noise = noise_level * np.random.normal(0, 1, Y.shape)
    Y += noise
    
    return Y


def run_inference_prediction(signal_loc_type, num_samples, time_array, time_span, kernel_coeff, target_fourier_coeff, noise, time_grid_points, alpha, **kwargs):
    """
    Runs the inference for both frequency-localized and time-localized signals and returns the prediction in the time domain.
    
    Parameters:
    ----------
    signal_loc_type : str
        Specifies whether to generate 'freq-loc' or 'time-loc' localized samples.
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
    time_grid_points : int
        Number of grid points.
    alpha : float
        Regularization exponent.

    
    Optional Parameters (kwargs):
    -----------------------------
    loc_parameter : float
        Time localization parameter (used for time-localized signals).
    freq_loc_inputs_decay : float
        Exponent for the power-law distribution (used for frequency-localized signals).
    freq_loc_max : int
        Maximum frequency value (used for frequency-localized signals).
    seed : int, optional
        Seed for random number generation.

    Returns:
    -------
    numpy.ndarray
        The prediction in the time domain (inverse Fourier of the predicted coefficients).
    """
    loc_parameter = kwargs.get('loc_parameter', None)  # Only used for time-localized signals
    freq_loc_inputs_decay = kwargs.get('freq_loc_inputs_decay', None)  # Used for frequency-localized signals
    freq_loc_max = kwargs.get('freq_loc_max', None)  # Maximum frequency for power-law sampling
    seed = kwargs.get('seed', None)  # Optional seed for reproducibility
    
    # Generate localized samples based on the signal_loc_type
    if signal_loc_type == 'time-loc':
        if loc_parameter is None:
            raise ValueError("Parameter 'loc_parameter' must be provided for time-localized signals.")
        X = generate_time_localized_samples(num_samples, time_array, loc_parameter)

    elif signal_loc_type == 'freq-loc':
        if freq_loc_inputs_decay is None or freq_loc_max is None:
            raise ValueError("Parameters 'freq_loc_inputs_decay' and 'freq_loc_max' must be provided for frequency-localized signals.")
        X = generate_frequency_localized_samples(num_samples, time_array, freq_loc_max, freq_loc_inputs_decay, power_law_samples, seed=seed)

    else:
        raise ValueError("Invalid signal_loc_type. Choose either 'time-loc' or 'freq-loc'.")
    
    # Compute Fourier coefficients of X and Y
    X_fourier = compute_fourier_coeff(X, time_span)
    Y = compute_output_data_matrix(X_fourier, target_fourier_coeff, noise, time_grid_points)
    Y_fourier = compute_fourier_coeff(Y, time_span)

    # Regularization and Fourier prediction
    lamb = 1e-4 * num_samples ** (-1 / (2 * alpha + 2))
    prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)

    for l in range(time_array.size):
        eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / num_samples
        term1 = kernel_coeff[l] / (eigenval + lamb)
        term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / num_samples
        prediction_fourier[l] = term1 * term2

    # Compute inverse FFT to get prediction in the time domain
    prediction_time_domain = np.fft.ifft(time_grid_points * prediction_fourier)
    
    return np.real(prediction_time_domain)




def run_inference_error(signal_loc_type, num_samples, num_experiments, time_array, time_span, kernel_coeff, target_fourier_coeff, noise, time_grid_points, alpha, series_truncation, **kwargs):
    """
    Runs the inference and error computation for both frequency-localized and time-localized signals.
    
    Parameters:
    ----------
    signal_type : str
        Specifies whether to generate 'freq-loc' or 'time-loc' localized samples.
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
    time_grid_points : int
        Number of grid points.
    alpha : float
        Regularization exponent.
    series_truncation : int
        Number of terms in the (kernel)series truncation for error computation.
    
    Optional Parameters (kwargs):
    -----------------------------
    loc_parameter : float
        Time localization parameter (used for time-localized signals).
    freq_loc_inputs_decay : float
        Exponent for the power-law distribution (used for frequency-localized signals).
    freq_loc_max : int
        Maximum frequency value (used for frequency-localized signals).
    seed : int, optional
        Seed for random number generation.

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
    error_sampmean = np.zeros(num_samples)
    error_sampstd = np.zeros(num_samples)
    error_logmean = np.zeros(num_samples)
    error_logstd = np.zeros(num_samples)
    
    loc_parameter = kwargs.get('loc_parameter', None)  # Only used for time-localized signals
    freq_loc_inputs_decay = kwargs.get('freq_loc_inputs_decay', None)      # Only used for frequency-localized signals
    freq_loc_max = kwargs.get('freq_loc_max', None)  # Maximum frequency value for power-law sampling
    seed = kwargs.get('seed', None)    # Optional seed for reproducibility
    
    for n in range(1, num_samples + 1):
        error_of_experiments = np.zeros(num_experiments)
        
        for j in range(num_experiments):
            if signal_loc_type == 'time-loc':
                if loc_parameter is None:
                    raise ValueError("Parameter 'loc_parameter' must be provided for time-localized signals.")
                
                # Generate time-localized samples
                X = generate_time_localized_samples(n, time_array, loc_parameter)
            
            elif signal_loc_type == 'freq-loc':
                if freq_loc_inputs_decay is None or freq_loc_max is None:
                    raise ValueError("Parameters 'freq_loc_inputs_decay' and 'freq_loc_max' must be provided for frequency-localized signals.")
                
                # Generate frequency-localized samples
                X = generate_frequency_localized_samples(n, time_array, freq_loc_max, freq_loc_inputs_decay, power_law_samples, seed=seed)
            
            else:
                raise ValueError("Invalid signal_type. Choose either 'time-loc' or 'freq-loc'.")

            # Compute Fourier coefficients of X and Y
            X_fourier = compute_fourier_coeff(X, time_span)
            Y = compute_output_data_matrix(X_fourier, target_fourier_coeff, noise, time_grid_points)
            Y_fourier = compute_fourier_coeff(Y, time_span)

            # Regularization and error computation
            lamb = 1e-4 * n ** (-1 / (2 * alpha + 2))
            prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)
            
            for l in range(time_array.size):
                eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / n
                term1 = kernel_coeff[l] / (eigenval + lamb)
                term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / n
                prediction_fourier[l] = term1 * term2

            # Compute the H error
            w_diff_coeff = prediction_fourier - target_fourier_coeff
            error_h_squared = (np.abs(w_diff_coeff[:series_truncation]) ** 2 / kernel_coeff[:series_truncation]).sum()
            error_of_experiments[j] = error_h_squared
        
        # Error statistics
        error_sampmean[n - 1] = np.mean(error_of_experiments)
        error_sampstd[n - 1] = np.std(error_of_experiments)
        error_logmean[n - 1] = np.mean(np.log(error_of_experiments))
        error_logstd[n - 1] = np.std(np.log(error_of_experiments))
    
    return error_sampmean, error_sampstd, error_logmean, error_logstd











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


