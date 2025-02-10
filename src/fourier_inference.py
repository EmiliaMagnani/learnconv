import numpy as np

from fourier import compute_fourier_coeff
from regularization import compute_lambda


def compute_output_data_matrix(
    X_fourier, target_f_coeff, noise_level, n_grid_points, time_interval
):
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
    Y = np.fft.ifft(target_f_coeff[:, np.newaxis] * X_fourier, axis=0) * (
        n_grid_points / time_interval
    )

    # Add noise to the entire matrix at once
    noise = noise_level * np.random.normal(0, 1, Y.shape)
    Y += noise

    return Y


def compute_prediction(
    num_samples, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X
):
    """
    Runs inference for time-localized signals and returns the prediction in the time domain.

    Parameters:
    ----------
    num_samples : int
        The number of samples used for the prediction.
    time_array : numpy.ndarray
        Array of time points.
    time_span : float
        Total time span.
    kernel_coeff : numpy.ndarray
        Array of kernel coefficients (in the Fourier domain) corresponding to the kernel function.
    target_coeff : numpy.ndarray
         Array of Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise added to the output.
    lamb : float
        The regularization parameter.
    X : np.ndarray
        The sample matrix (time-domain) with shape (len(time_array), num_samples).

    Returns:
    -------
    prediction_fourier : np.ndarray
        The predicted Fourier coefficients (a 1D array of length equal to len(time_array)).
    prediction_time_domain : np.ndarray
        The time-domain prediction obtained by applying the inverse Fourier transform to
        prediction_fourier.
    """

    # Compute Fourier coefficients of X and Y
    X_fourier = compute_fourier_coeff(X, time_span)
    Y = compute_output_data_matrix(
        X_fourier, target_coeff, noise, time_array.size, time_span
    )
    Y_fourier = compute_fourier_coeff(Y, time_span)

    eigenvals = kernel_coeff * np.sum(np.abs(X_fourier) ** 2, axis=1) / num_samples
    term1 = kernel_coeff / (eigenvals + lamb)
    term2 = np.sum(np.conj(X_fourier) * Y_fourier, axis=1) / num_samples
    prediction_fourier = term1 * term2
    prediction_time_domain = np.fft.ifft(prediction_fourier) * (
        time_array.size / time_span
    )

    # prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)
    # for l in range(time_array.size):
    #     eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / num_samples
    #     term1 = kernel_coeff[l] / (eigenval + lamb)
    #     term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / num_samples
    #     prediction_fourier[l] =  (term1 * term2)

    return prediction_fourier, np.real(prediction_time_domain)


def compute_error(
    num_samples,
    num_experiments,
    time_array,
    time_span,
    kernel_coeff,
    target_coeff,
    noise,
    r,
    b,
    const,
    sample_generator,
    sample_gen_params,
):
    """
    Computes the squared error in the reproducing kernel Hilbert space (RKHS) norm for an
    inference method by averaging over multiple experiments for increasing sample sizes. This
    generic function uses a provided sample generation routine so that it can be used with
    different types of input signals (e.g., time-localized or frequency-localized).

    The RKHS error is computed as:

        error² = sum [|prediction_fourier - target_coeff|² / kernel_coeff],

    where the prediction is obtained by performing a regularized inversion in the Fourier
    domain. The regularization parameter λ is computed as:

        λ = const * n^( -1 / (2*r + 1 + 1/b) )

    Parameters
    ----------
    num_samples : int
        The maximum number of samples (n) to test. The function computes errors for each n
        from 1 to num_samples.
    num_experiments : int
        The number of independent experiments to average over for each sample size.
    time_array : np.ndarray
        The array of time points defining the signal domain.
    time_span : float
        The total length of the time interval (e.g., the period of the signal).
    kernel_coeff : np.ndarray
        Array of kernel coefficients (in Fourier space) used in the eigenvalue problem.
    target_coeff : np.ndarray
        The true Fourier coefficients of the target function.
    noise : float
        The standard deviation of the noise added to the output data.
    r : float
        The regularization exponent that influences the decay rate of the regularization parameter.
    b : float
        The source condition parameter used in the regularization.
    const : float
        A constant factor used to compute the regularization parameter λ.
    sample_generator : callable
        Function used to generate the sample matrix X. It must have the signature:

            X = sample_generator(n, time_array, **sample_gen_params)

        For example, for time-localized samples this might be a function that returns
        a matrix of shape (len(time_array), n) based on a localization parameter.
    sample_gen_params : dict
        A dictionary of additional parameters required by the sample_generator function. For
        instance, for time-localized samples you might pass {'delta': loc_parameter}, and for
        frequency-localized samples you might pass {'max_value': freq_loc_max, 'exponent': freq_loc_inputs_decay,
        'rng': np.random.default_rng(seed)}.

    Returns
    -------
    error_sampmean : np.ndarray
        A 1D array of length num_samples containing the SQUARED mean RKHS error for each sample size.
    error_sampstd : np.ndarray
        A 1D array of length num_samples containing the standard deviation of the RKHS error for each sample size.
    """
    error_sampmean = np.zeros(num_samples)
    error_sampstd = np.zeros(num_samples)
    # error_logmean = np.zeros(num_samples)  # Optional: log-scale means
    # error_logstd = np.zeros(num_samples)   # Optional: log-scale standard deviations

    for n in range(1, num_samples + 1):
        errors = np.zeros(num_experiments)
        for j in range(num_experiments):
            # Generate the sample matrix X using the provided sample generator
            X = sample_generator(n, time_array, **sample_gen_params)

            # Compute the regularization parameter λ
            lamb = compute_lambda(const, n, r, b)

            # Use compute_prediction to get the Fourier prediction
            prediction_fourier, _ = compute_prediction(
                n, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X
            )

            # Compute the RKHS error:
            diff = prediction_fourier - target_coeff
            error = np.sum(np.abs(diff) ** 2 / kernel_coeff)
            errors[j] = error  # it's the SQUARED mean squared error

        error_sampmean[n - 1] = np.mean(errors)
        error_sampstd[n - 1] = np.std(errors)
        # error_logmean[n - 1] = np.mean(np.log(errors))
        # error_logstd[n - 1] = np.std(np.log(errors))

    return error_sampmean, error_sampstd


def compute_error_grid_search_over_lambda(
    num_samples,
    num_experiments,
    time_array,
    time_span,
    kernel_coeff,
    target_coeff,
    noise,
    sample_generator,
    sample_gen_params,
    lambda_candidates,
):
    """
    Computes the RKHS error for each sample size n, where for each n the optimal lambda is chosen
    via grid search over lambda_candidates.

    Parameters:
    -----------
    num_samples : int
        Maximum number of samples. The function computes errors for each n from 1 to num_samples.
    num_experiments : int
        Number of independent experiments per sample size.
    time_array : np.ndarray
        Array of time points.
    time_span : float
        Length of the time interval (e.g., the period of the signal).
    kernel_coeff : np.ndarray
        Kernel coefficients (Fourier domain).
    target_coeff : np.ndarray
        Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise added to the output data
    sample_generator : callable
        Function to generate sample matrices.
    sample_gen_params : dict
        Additional parameters for the sample_generator.
    lambda_candidates : array-like
        A list or array of candidate lambda values to search over.

    Returns:
    --------
    error_sampmean : np.ndarray
        A 1D array of length num_samples containing the SQUARED mean RKHS error for each sample size with lamda optimized
    error_sampstd : np.ndarray
        A 1D array of length num_samples containing the standard deviation of the RKHS error for each sample size with lambda optimized.
    """
    error_sampmean = np.zeros(num_samples)
    error_sampstd = np.zeros(num_samples)

    # Loop over sample sizes n = 1 to num_samples.
    for n in range(1, num_samples + 1):
        errors = np.zeros(num_experiments)
        for j in range(num_experiments):
            # Generate sample matrix X for the current experiment.
            X = sample_generator(n, time_array, **sample_gen_params)

            best_error = np.inf
            best_lambda = None

            # Grid search over lambda candidates.
            for lamb in lambda_candidates:
                # Get the prediction using the candidate lambda.
                prediction_fourier, _ = compute_prediction(
                    n, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X
                )
                # Compute the RKHS error.
                diff = prediction_fourier - target_coeff
                current_error = np.sum(np.abs(diff) ** 2 / kernel_coeff)

                if current_error < best_error:
                    best_error = current_error
                    best_lambda = lamb

            # Optionally, you can log or store best_lambda for each run.
            errors[j] = best_error

        error_sampmean[n - 1] = np.mean(errors)
        error_sampstd[n - 1] = np.std(errors)

    return error_sampmean, error_sampstd


############################################################################################################################

# def run_inference_time_loc(num_samples, time_array, time_span, kernel_coeff, target_coeff, noise, r, b,const, loc_parameter):
#     """
#     Runs inference for time-localized signals and returns the prediction in the time domain.

#     Parameters:
#     ----------
#     num_samples : int
#         Number of input samples for inference.
#     time_array : numpy.ndarray
#         Array of time points.
#     time_span : float
#         Total time span.
#     kernel_coeff : numpy.ndarray
#         Kernel coefficients used in the eigenvalue problem.
#     target_coeff : numpy.ndarray
#          Fourier coefficients of the target function.
#     noise : float
#         Standard deviation of the noise added to the output.
#     r : float
#         Regularization exponent.
#     b : float
#         Source conditions parameter.
#     const : float
#         Constant factor for the regularization \lambda
#     loc_parameter : float
#         Time localization parameter for generating time-localized samples.

#     Returns:
#     -------
#     numpy.ndarray
#         The prediction in the time domain (inverse Fourier of the predicted coefficients).
#     """
#     # Generate time-localized samples
#     X = generate_time_localized_samples(num_samples, time_array, loc_parameter)


#     # Compute Fourier coefficients of X and Y
#     X_fourier = compute_fourier_coeff(X, time_span)
#     Y = compute_output_data_matrix(X_fourier, target_coeff, noise, time_array.size, time_span)
#     Y_fourier = compute_fourier_coeff(Y, time_span)

#     # Regularization and Fourier prediction
#     lamb =  const* num_samples ** (-1 / (2 * r + 1 + 1/b))   #1e-4 *
#     # prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)

#     # for l in range(time_array.size):
#     #     eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / num_samples
#     #     term1 = kernel_coeff[l] / (eigenval + lamb)
#     #     term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / num_samples
#     #     prediction_fourier[l] =  (term1 * term2)

#     ## vectorized version of the above loop
#     eigenvals = kernel_coeff * np.sum(np.abs(X_fourier) ** 2, axis=1) / num_samples
#     term1 = kernel_coeff / (eigenvals + lamb)
#     term2 = np.sum(np.conj(X_fourier) * Y_fourier, axis=1) / num_samples
#     prediction_fourier = term1 * term2

#     # Compute inverse FFT to get prediction in the time domain
#     prediction_time_domain = np.fft.ifft(prediction_fourier) * (time_array.size / time_span)

#     return prediction_fourier, np.real(prediction_time_domain)

# def run_inference_freq_loc(num_samples, time_array, time_span, kernel_coeff, target_coeff, noise, r, b, const, freq_loc_inputs_decay, freq_max, seed=None):
#     """
#     Runs inference for frequency-localized signals and returns the prediction in the time domain.

#     Parameters:
#     ----------
#     num_samples : int
#         Number of samples for inference.
#     time_array : numpy.ndarray
#         Array of time points.
#     time_span : float
#         Total time span.
#     kernel_coeff : numpy.ndarray
#         Kernel coefficients used in the eigenvalue problem.
#     target_coeff : numpy.ndarray
#          Fourier coefficients of the target function.
#     noise : float
#         Standard deviation of the noise added to the output.
#     r : float
#         Regularization exponent.
#     b : float
#         Source conditions parameter.
#     const : float
#         Constant factor for the regularization \lambda
#     freq_loc_inputs_decay : float
#         Exponent for the power-law distribution.
#     freq_max : int
#         Maximum frequency value for generating frequency-localized samples.
#     seed : int, optional
#         Seed for random number generation (default is None).

#     Returns:
#     -------
#     numpy.ndarray, numpy.ndarray
#         the Fourier coefficients of the prediction and the The prediction in the time domain (inverse Fourier of the predicted coefficients).
#     """
#     # Generate frequency-localized samples
#     X = generate_frequency_localized_samples(num_samples, time_array, freq_max, freq_loc_inputs_decay, power_law_samples_symmetric, seed=seed)

#     # Compute Fourier coefficients of X and Y
#     X_fourier = compute_fourier_coeff(X, time_span)
#     Y = compute_output_data_matrix(X_fourier, target_coeff, noise, time_array.size, time_span)
#     Y_fourier = compute_fourier_coeff(Y, time_span)

#     # Regularization and Fourier prediction
#     lamb = const * num_samples ** (-1 / (2 * r + 1 + 1/b))

#     # prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)
#     # for l in range(time_array.size):
#     #     eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / num_samples
#     #     term1 = kernel_coeff[l] / (eigenval + lamb)
#     #     term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / num_samples
#     #     prediction_fourier[l] = (term1 * term2)   # times 2 factor if you cover only half of the frequencies (positive frequencies, power_law)?

#     ## vectorized version of the above loop
#     eigenvals = kernel_coeff * np.sum(np.abs(X_fourier) ** 2, axis=1) / num_samples
#     term1 = kernel_coeff / (eigenvals + lamb)
#     term2 = np.sum(np.conj(X_fourier) * Y_fourier, axis=1) / num_samples
#     prediction_fourier = term1 * term2


#     # Compute inverse FFT to get prediction in the time domain
#     prediction_time_domain = np.fft.ifft(prediction_fourier) * (time_array.size / time_span)
#     # fft_pred_fourier_coeff = compute_fourier_coeff(prediction_time_domain, time_span) # to check wether it is the same as prediction_fourier_coeff

#     return  prediction_fourier, np.real(prediction_time_domain)


# def compute_error_time_loc(num_samples, num_experiments, time_array, time_span, kernel_coeff, target_coeff, noise, r, b, const, loc_parameter):
#     """
#     Runs the inference and error computation for time-localized signals.

#     Parameters:
#     ----------
#     num_samples : int
#         Number of samples for inference.
#     num_experiments : inti
#         Number of experiments to average over.
#     time_array : numpy.ndarray
#         Array of time points.
#     time_span : float
#         Total time span.
#     kernel_coeff : numpy.ndarray
#         Kernel coefficients used in the eigenvalue problem.
#     target_coeff : numpy.ndarray
#          Fourier coefficients of the target function.
#     noise : float
#         Standard deviation of the noise added to the output.
#     r : float
#         Regularization exponent.
#     b : float
#         Source conditions parameter.
#     const : float
#         Constant factor for the regularization \lambda
#     loc_parameter : float
#         Time localization parameter for generating time-localized samples.

#     Returns:
#     -------
#     error_sampmean : numpy.ndarray
#         Mean error for each sample.
#     error_sampstd : numpy.ndarray
#         Standard deviation of error for each sample.
#     """
#     error_squared_sampmean = np.zeros(num_samples)
#     error_squared_sampstd = np.zeros(num_samples)
#     #you can also return these if needed
#     error_squared_logmean = np.zeros(num_samples)
#     error_squared_logstd = np.zeros(num_samples)

#     for n in range(1, num_samples + 1):
#         error_of_experiments = np.zeros(num_experiments)

#         for j in range(num_experiments):
#             # Generate time-localized samples
#             X = generate_time_localized_samples(n, time_array, loc_parameter)

#             # Compute Fourier coefficients of X and Y
#             X_fourier = compute_fourier_coeff(X, time_span)
#             Y = compute_output_data_matrix(X_fourier, target_coeff, noise, time_array.size, time_span)
#             Y_fourier = compute_fourier_coeff(Y, time_span)

#             # Regularization and error computation
#             lamb = const* n ** (-1 / (2 * r + 1 + 1/b))   #1e-4 *
#             prediction_fourier = np.zeros(time_array.size, dtype=np.complex128)

#             for l in range(time_array.size):
#                 eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / n
#                 term1 = kernel_coeff[l] / (eigenval + lamb)
#                 term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / n
#                 prediction_fourier[l] = term1 * term2


#             # Compute the H error
#             w_diff_coeff = prediction_fourier - target_coeff  #there was abs in both terms. why?
#             error_h_squared = (np.abs(w_diff_coeff) ** 2 / kernel_coeff).sum()
#             error_of_experiments[j] = error_h_squared

#         # Error statistics
#         error_squared_sampmean[n - 1] = np.mean(error_of_experiments)
#         error_squared_sampstd[n - 1] = np.std(error_of_experiments)
#         #you can also return these if needed
#         error_squared_logmean[n - 1] = np.mean(np.log(error_of_experiments))
#         error_squared_logstd[n - 1] = np.std(np.log(error_of_experiments))

#     return  error_squared_sampmean, error_squared_sampstd


# def compute_error_freq_loc(num_samples, num_experiments, time_array, time_span, kernel_coeff, target_coeff, noise,  r, b, const, freq_loc_inputs_decay, freq_loc_max, seed=None):
#     """
#     Runs the inference and error computation for frequency-localized signals.

#     Parameters:
#     ----------
#     num_samples : int
#         Number of samples for inference.
#     num_experiments : int
#         Number of experiments to average over.
#     time_array : numpy.ndarray
#         Array of time points.
#     time_span : float
#         Total time span.
#     kernel_coeff : numpy.ndarray
#         Kernel coefficients used in the eigenvalue problem.
#     target_coeff : numpy.ndarray
#          Fourier coefficients of the target function.
#     noise : float
#         Standard deviation of the noise added to the output.
#     r : float
#         Regularization exponent.
#     b : float
#         Source conditions parameter.
#     const : float
#         Constant factor for the regularization \lambda
#     freq_loc_inputs_decay : float
#         Exponent for the power-law distribution (used for frequency-localized signals).
#     freq_loc_max : int
#         Maximum frequency value for generating frequency-localized samples.
#     seed : int, optional
#         Seed for random number generation (default is None).

#     Returns:
#     -------
#     error_sampmean : numpy.ndarray
#         Mean error for each sample.
#     error_sampstd : numpy.ndarray
#         Standard deviation of error for each sample.
#     """
#     error_squared_sampmean = np.zeros(num_samples)
#     error_squared_sampstd = np.zeros(num_samples)
#     #you can also return these if needed:
#     error_squared_logmean = np.zeros(num_samples)
#     error_squared_logstd = np.zeros(num_samples)

#     for n in range(1, num_samples + 1):
#         error_squared_exp = np.zeros(num_experiments)

#         for j in range(num_experiments):
#             # Generate frequency-localized samples
#             X = generate_frequency_localized_samples(n, time_array, freq_loc_max, freq_loc_inputs_decay, power_law_samples, seed=seed)

#             # Compute Fourier coefficients of X and Y
#             X_fourier = compute_fourier_coeff(X, time_span)
#             Y = compute_output_data_matrix(X_fourier, target_coeff, noise, time_array.size, time_span)
#             Y_fourier = compute_fourier_coeff(Y, time_span)

#             # Regularization and error computation
#             lamb = const * n ** (-1 / (2 * r + 1 + 1/b))   # 1e-4*
#             prediction_fourier_coeff = np.zeros(time_array.size, dtype=np.complex128)

#             for l in range(time_array.size):
#                 eigenval = kernel_coeff[l] * (np.abs(X_fourier[l, :]) ** 2).sum() / n
#                 term1 = kernel_coeff[l] / (eigenval + lamb)
#                 term2 = (np.conj(X_fourier[l, :]) * Y_fourier[l, :]).sum() / n
#                 prediction_fourier_coeff[l] = (term1 * term2)   # times 2 factor if you cover only half of the frequencies (positive frequencies, power_law)?

#             # prediction_time_domain = np.fft.ifft(prediction_fourier_coeff) * (time_array.size / time_span)
#             #fft_pred_fourier_coeff = compute_fourier_coeff(prediction_time_domain, time_span) # use this below and checked wether it is the same as prediction_fourier_coeff

#             # Compute the error in H-norm
#             w_diff_coeff = prediction_fourier_coeff - target_coeff    #abs in both terms before
#             error_h_squared = (np.abs(w_diff_coeff) ** 2 / kernel_coeff).sum()
#             error_squared_exp[j] = error_h_squared

#         # Error statistics
#         error_squared_sampmean[n - 1] = np.mean(error_squared_exp)
#         error_squared_sampstd[n - 1] = np.std(error_squared_exp)
#         #you can also return these if needed
#         error_squared_logmean[n - 1] = np.mean(np.log(error_squared_exp))
#         error_squared_logstd[n - 1] = np.std(np.log(error_squared_exp))

#     return  error_squared_sampmean, error_squared_sampstd
