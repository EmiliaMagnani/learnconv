import numpy as np
from typing import Optional, Tuple
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


def _compute_prediction_given_lambda(
    num_samples, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X_fourier
):
    """
    Runs inference (in Fourier domain) for input signals X and returns the Fourier coefficients of the prediction, and the prediction in the time domain,
    for a given regularization parameter lambda.

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
    X_fourier : np.ndarray
        Fourier coefficients of the input matrix with size (len(time_array), num_samples).

    Returns:
    -------
    prediction_fourier : np.ndarray
        The predicted Fourier coefficients (a 1D array of length equal to len(time_array)).
    prediction_time_domain : np.ndarray
        The time-domain prediction obtained by applying the inverse Fourier transform to
        prediction_fourier.
    """

    # Compute the output data matrix Y
    Y = compute_output_data_matrix(
        X_fourier, target_coeff, noise, time_array.size, time_span
    )
    Y_fourier = compute_fourier_coeff(Y, time_span)

    eigenvals = kernel_coeff * np.sum(np.abs(X_fourier) ** 2, axis=1) / num_samples
    term1 = kernel_coeff / (eigenvals + lamb + 1e-8) # Add a small value to avoid division by zero
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


def compute_prediction(
    num_samples: int,
    time_array: np.ndarray,
    time_span: float,
    kernel_coeff: np.ndarray,
    target_coeff: np.ndarray,
    noise: float,
    X_fourier: np.ndarray,
    lamb: Optional[float] = None,
    optimize_lambda: bool = False,
    lambda_candidates: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference (in the Fourier domain) for input signals X and returns both the Fourier
    coefficients of the prediction and the prediction in the time domain.
    
    Optionally, performs a grid search over lambda if optimize_lambda is True. In that case,
    the function selects the lambda candidate that minimizes the RKHS error:
    
        error = sum(|prediction_fourier - target_coeff|^2 / kernel_coeff).
    
    When grid search is enabled, if no candidate array is provided, candidates are computed using
    a logspace interval based on the maximum scale of the problem:
    
        sigma_max = max_l { kernel_coeff[l] * (1/num_samples)*sum_n |X_fourier[l, n]|^2 }
        lambda_candidates = sigma_max * logspace(-k_min, -k_max, num_candidates)
    
    Parameters
    ----------
    num_samples : int
        The number of samples used for the prediction.
    time_array : np.ndarray
        Array of time points.
    time_span : float
        Total time span.
    kernel_coeff : np.ndarray
        Array of kernel coefficients (in Fourier space).
    target_coeff : np.ndarray
        Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise added to the output.
    X_fourier : np.ndarray
        Fourier coefficients of the input matrix. Size is (len(time_array), num_samples).
    lamb : float, optional
        The regularization parameter (required if optimize_lambda is False).
    optimize_lambda : bool, default False
        If True, perform a grid search over lambda.
    lambda_candidates : array-like, optional
        An array of candidate lambda values. If None and optimize_lambda is True, the candidates are
        computed dynamically using logspace.
    
    Returns
    -------
    prediction_fourier : np.ndarray
        The predicted Fourier coefficients.
    prediction_time_domain : np.ndarray
        The prediction in the time domain.
    
    Raises
    ------
    ValueError
        If optimize_lambda is True and a fixed lambda is provided, or if fixed lambda is requested
        but not provided.
    """
    if optimize_lambda:
        if lamb is not None:
            raise ValueError(
                "When optimize_lambda is True, 'lamb' must not be provided. "
                "Lambda will be optimized via grid search."
            )
        # Compute the per-frequency sigma: average squared magnitude times kernel_coeff.
        sigma_values = kernel_coeff * np.mean(np.abs(X_fourier) ** 2, axis=1)
        sigma_max = np.max(sigma_values)
        
        # Define the candidate search in logspace:
        k_min = 3  # exponent lower bound (e.g., sigma_max*10^-3)  #for heat equation put lower, 4 or 5
        k_max = 1  # exponent upper bound (e.g., sigma_max*10^-1)
        num_candidates = 35  # Adjust as needed
        lambda_candidates = sigma_max * np.logspace(-k_min, -k_max, num=num_candidates)
        
        print(f"Computed sigma_max = {sigma_max:.3e}")
        print(f"Lambda candidates range: from {sigma_max * 10**(-k_min):.3e} to {sigma_max * 10**(-k_max):.3e}")

        # Initialize with the candidate corresponding to the smallest lambda.
        best_lambda = lambda_candidates[0]
        best_pred_fourier, best_pred_time = _compute_prediction_given_lambda(
            num_samples, time_array, time_span, kernel_coeff, target_coeff, noise, best_lambda, X_fourier
        )
        best_error = np.sum(np.abs(best_pred_fourier - target_coeff) ** 2 / kernel_coeff)
        
        print(f"Initial error with lambda = {best_lambda:.3e}: {best_error:.3e}")

        # Grid search over candidate lambda values.
        for candidate in lambda_candidates:
            pred_fourier, pred_time = _compute_prediction_given_lambda(
                num_samples, time_array, time_span, kernel_coeff, target_coeff, noise, candidate, X_fourier
            )
            error = np.sum(np.abs(pred_fourier - target_coeff) ** 2 / kernel_coeff)
            # print(f"Candidate lambda = {candidate:.3e} has error = {error:.3e}")
            if error < best_error:
                best_error = error
                best_lambda = candidate
                best_pred_fourier = pred_fourier
                best_pred_time = pred_time

        print(f"Best lambda: {best_lambda:.3e} with error: {best_error:.3e}")
        return best_pred_fourier, best_pred_time

    else:
        if lamb is None:
            raise ValueError("Parameter 'lamb' must be provided when optimize_lambda is False.")
        return _compute_prediction_given_lambda(
            num_samples, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X_fourier
        )




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
    optimize_lambda=False,
    lambda_candidates=None
):
    """
    Computes the squared RKHS error for the inference method by averaging over multiple experiments,
    for increasing sample sizes. This function allows you to decide whether to use grid search
    to optimize lambda or to use the regularization parameter computed by compute_lambda.

    The RKHS error is computed as:
    
        error² = sum (|prediction_fourier - target_coeff|² / kernel_coeff)

    Parameters
    ----------
    num_samples : int
        Maximum number of samples (n). Errors are computed for each n = 1, 2, ..., num_samples.
    num_experiments : int
        Number of independent experiments to average over for each sample size.
    time_array : numpy.ndarray
        Array of time points.
    time_span : float
        Total time interval (e.g., the period of the signal).
    kernel_coeff : numpy.ndarray
        Kernel coefficients (in Fourier space).
    target_coeff : numpy.ndarray
        Fourier coefficients of the target function.
    noise : float
        Standard deviation of the noise.
    r : float
        The regularization exponent.
    b : float
        The source condition parameter.
    const : float
        Constant used to compute the regularization parameter λ (when not optimized).
    sample_generator : callable
        Function used to generate the sample matrix X. Must have signature:
            X = sample_generator(n, time_array, **sample_gen_params)
    sample_gen_params : dict
        Additional parameters for sample_generator.
    optimize_lambda : bool, default False
        If True, grid search will be performed inside compute_prediction.
    lambda_candidates : array-like, optional
        A list/array of candidate lambda values. If None and optimize_lambda is True,
        compute_prediction will compute the candidates dynamically.

    Returns
    -------
    error_sampmean : numpy.ndarray
        1D array (length=num_samples) of the mean squared RKHS error for each sample size.
    error_sampstd : numpy.ndarray
        1D array (length=num_samples) of the standard deviation of the RKHS error for each sample size.
    """
    error_sampmean = np.zeros(num_samples)
    error_sampstd = np.zeros(num_samples)

    for n in range(1, num_samples + 1):
        errors = np.zeros(num_experiments)
        for j in range(num_experiments):
            # Generate the sample matrix X for the current experiment.
            X = sample_generator(n, time_array, **sample_gen_params)
            # Compute Fourier coefficients of X.
            X_fourier = compute_fourier_coeff(X, time_span)

            if optimize_lambda:
                # Grid search inside compute_prediction.
                # If lambda_candidates is None, compute_prediction will generate them.
                prediction_fourier, _ = compute_prediction(
                    n, time_array, time_span, kernel_coeff, target_coeff, noise, X_fourier,
                    optimize_lambda=True, lambda_candidates=lambda_candidates
                )
            else:
                # Compute lambda using the provided parameters.
                lamb = compute_lambda(const, n, r, b)
                prediction_fourier, _ = compute_prediction(
                    n, time_array, time_span, kernel_coeff, target_coeff, noise, X_fourier, lamb=lamb
                )

            # Compute the RKHS error.
            diff = prediction_fourier - target_coeff
            error = np.sum(np.abs(diff) ** 2 / kernel_coeff)
            errors[j] = error

        error_sampmean[n - 1] = np.mean(errors)
        error_sampstd[n - 1] = np.std(errors)

    return error_sampmean, error_sampstd


# def compute_operator_error(
#     num_samples,
#     num_experiments,
#     time_array,
#     time_span,
#     kernel_coeff,
#     target_coeff,
#     noise,
#     r,
#     b,
#     const,
#     sample_generator,
#     sample_gen_params,
#     optimize_lambda=False,
#     lambda_candidates=None
# ):
#     op_error_sampmean = np.zeros(num_samples)
#     op_error_sampstd = np.zeros(num_samples)
    
#     # Determine whether to restrict the frequencies:
#     # Here we check if the sample generator is the time-localized one by
#     # looking for a 'delta' key in sample_gen_params.
#     if "delta" in sample_gen_params:
#         delta = sample_gen_params["delta"]
#         grid_size = len(time_array)
#         freqs = np.fft.fftfreq(grid_size, d=(time_span / grid_size))
#         ell_max = min(grid_size // 2, int(1 / (10 * delta)), int(0.1 / (2 * np.pi * delta)))
#         freq_mask = np.abs(freqs) <= ell_max
#         # Restrict kernel_coeff to the same low-frequency band.
#         kernel_coeff_masked = kernel_coeff[freq_mask]
#     else:
#         freq_mask = None
#         kernel_coeff_masked = kernel_coeff

#     for n in range(1, num_samples + 1):
#         errors = np.zeros(num_experiments)
#         for j in range(num_experiments):
#             # Generate the sample matrix X.
#             X = sample_generator(n, time_array, **sample_gen_params)
            
#             # Compute Fourier coefficients of X.
#             X_fourier = compute_fourier_coeff(X, time_span)
#             # Compute average power per frequency.
#             avg_power = np.mean(np.abs(X_fourier) ** 2, axis=1)
#             # If we have a frequency mask (for time-localized inputs), restrict avg_power.
#             if freq_mask is not None:
#                 avg_power = avg_power[freq_mask]

            
#             if optimize_lambda:
#                 # (Grid search to select lambda.)
#                 sigma_max = np.max(avg_power * kernel_coeff_masked)
#                 sigma_max = sigma_max if sigma_max > 1e-12 else 1e-12
#                 k_min = 3  
#                 k_max = 1  
#                 num_candidates = 35
#                 lambda_candidates = sigma_max * np.logspace(-k_min, -k_max, num=num_candidates)

#                 best_lambda = None
#                 best_error = np.inf
#                 for candidate in lambda_candidates:
#                     pred_fourier, _ = _compute_prediction_given_lambda(
#                         n, time_array, time_span, kernel_coeff, target_coeff, noise, candidate, X_fourier
#                     )
#                     # Compute difference over all frequencies.
#                     diff = pred_fourier - target_coeff
#                     # Restrict diff if necessary.
#                     if freq_mask is not None:
#                         diff = diff[freq_mask]
#                     op_error = np.sum(avg_power * np.abs(diff) ** 2)
#                     if op_error < best_error:
#                         best_error = op_error
#                         best_lambda = candidate
#                         best_pred_fourier = pred_fourier
#                 prediction_fourier = best_pred_fourier
#             else:
#                 lamb = compute_lambda(const, n, r, b)
#                 prediction_fourier, _ = _compute_prediction_given_lambda(
#                     n, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X_fourier
#                 )
            
#             # Compute the difference and restrict frequencies if needed.
#             diff = prediction_fourier - target_coeff
#             if freq_mask is not None:
#                 diff = diff[freq_mask]
            
#             op_error = np.sum(avg_power * np.abs(diff) ** 2)
#             errors[j] = op_error

#         op_error_sampmean[n - 1] = np.mean(errors)
#         op_error_sampstd[n - 1] = np.std(errors)

#     return op_error_sampmean, op_error_sampstd

def compute_operator_error(
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
    optimize_lambda=False,
    lambda_candidates=None
):
    op_error_sampmean = np.zeros(num_samples)
    op_error_sampstd = np.zeros(num_samples)
    
    # Determine whether to restrict the frequencies (for time-localized inputs)
    if "delta" in sample_gen_params:
        delta = sample_gen_params["delta"]
        grid_size = len(time_array)
        freqs = np.fft.fftfreq(grid_size, d=(time_span / grid_size))
        # Define ell_max such that only frequencies |ℓ| << 1/δ are used.
        ell_max = min(grid_size // 2, int(1 / (10 * delta)), int(0.1 / (2 * np.pi * delta)))
        freq_mask = np.abs(freqs) <= ell_max
        # Restrict kernel_coeff to the same frequency range.
        kernel_coeff_masked = kernel_coeff[freq_mask]
    else:
        freq_mask = None
        kernel_coeff_masked = kernel_coeff

    for n in range(1, num_samples + 1):
        errors = np.zeros(num_experiments)
        for j in range(num_experiments):
            # Generate the sample matrix X.
            X = sample_generator(n, time_array, **sample_gen_params)
            
            # Compute Fourier coefficients of X.
            X_fourier = compute_fourier_coeff(X, time_span)
            # Compute average power per frequency.
            avg_power = np.mean(np.abs(X_fourier) ** 2, axis=1)
            if freq_mask is not None:
                avg_power = avg_power[freq_mask]
            
            if optimize_lambda:
                # Use the masked kernel coefficients here.
                sigma_max = np.max(avg_power * kernel_coeff_masked)
                sigma_max = sigma_max if sigma_max > 1e-12 else 1e-12
                k_min = 3  
                k_max = 1  
                num_candidates = 35
                lambda_candidates = sigma_max * np.logspace(-k_min, -k_max, num=num_candidates)

                best_lambda = None
                best_error = np.inf
                for candidate in lambda_candidates:
                    pred_fourier, _ = _compute_prediction_given_lambda(
                        n, time_array, time_span, kernel_coeff, target_coeff, noise, candidate, X_fourier
                    )
                    diff = pred_fourier - target_coeff
                    if freq_mask is not None:
                        diff = diff[freq_mask]
                    op_error = np.sum(avg_power * np.abs(diff) ** 2)
                    if op_error < best_error:
                        best_error = op_error
                        best_lambda = candidate
                        best_pred_fourier = pred_fourier
                prediction_fourier = best_pred_fourier
            else:
                lamb = compute_lambda(const, n, r, b)
                prediction_fourier, _ = _compute_prediction_given_lambda(
                    n, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X_fourier
                )
            
            diff = prediction_fourier - target_coeff
            if freq_mask is not None:
                diff = diff[freq_mask]
            op_error = np.sum(avg_power * np.abs(diff) ** 2)
            errors[j] = op_error

        op_error_sampmean[n - 1] = np.mean(errors)
        op_error_sampstd[n - 1] = np.std(errors)

    return op_error_sampmean, op_error_sampstd











# def compute_operator_error(
#     num_samples,
#     num_experiments,
#     time_array,
#     time_span,
#     kernel_coeff,
#     target_coeff,
#     noise,
#     r,
#     b,
#     const,
#     sample_generator,
#     sample_gen_params,
#     optimize_lambda=False,
#     lambda_candidates=None
# ):
#     """
#     Computes the squared operator error
#       ||Σ^(1/2)(w_n^λ - w_*)||²_H = ∑_{ξ} [\sigma * |(Fw_n^λ)_ξ - (Fw_*)_ξ|² / kernel_coeff[ξ]]
#     averaged over multiple experiments for sample sizes n = 1, 2, ..., num_samples.
    
#     Here, \sigma is estimated by:
#          sigma_est[l] = kernel_coeff[l] * (1/n)*sum_{i=1}^n |(FX)_l(i)|^2.
    
#     If optimize_lambda is True, the lambda is optimized using the operator error metric.
    
#     Parameters
#     ----------
#     num_samples : int
#         Maximum number of samples (n). Errors are computed for each n = 1, 2, ..., num_samples.
#     num_experiments : int
#         Number of independent experiments to average over for each sample size.
#     time_array : numpy.ndarray
#         Array of time points.
#     time_span : float
#         Total time interval (e.g., the period of the signal).
#     kernel_coeff : numpy.ndarray
#         Fourier coefficients of the kernel (i.e. the \hat{K}_ξ values).
#     target_coeff : numpy.ndarray
#         Fourier coefficients of the target function w_*.
#     noise : float
#         Standard deviation of the noise.
#     r : float
#         The regularization exponent.
#     b : float
#         The source condition parameter.
#     const : float
#         Constant used to compute the regularization parameter λ (when not optimized).
#     sample_generator : callable
#         Function used to generate the sample matrix X. Must have signature:
#             X = sample_generator(n, time_array, **sample_gen_params)
#     sample_gen_params : dict
#         Additional parameters for sample_generator.
#     optimize_lambda : bool, default False
#         If True, grid search will be performed to optimize λ using the operator error metric.
#     lambda_candidates : array-like, optional
#         Candidate λ values. If None and optimize_lambda is True, candidates are computed automatically.
    
#     Returns
#     -------
#     op_error_sampmean : numpy.ndarray
#         1D array (length=num_samples) of the mean squared operator error for each sample size.
#     op_error_sampstd : numpy.ndarray
#         1D array (length=num_samples) of the standard deviation of the operator error for each sample size.
#     """
#     op_error_sampmean = np.zeros(num_samples)
#     op_error_sampstd = np.zeros(num_samples)

#     for n in range(1, num_samples + 1):
#         errors = np.zeros(num_experiments)
#         for j in range(num_experiments):
#             # Generate the sample matrix X.
#             X = sample_generator(n, time_array, **sample_gen_params)
            
#             # Compute Fourier coefficients of X.
#             X_fourier = compute_fourier_coeff(X, time_span)
#             # Empirical estimate of sigma_l:
#             avg_power =  np.mean(np.abs(X_fourier) ** 2, axis=1)   #shape (len(time_array),)  
#             sigma_est = kernel_coeff * avg_power

#             if optimize_lambda:
#                 # If no lambda_candidates are provided, define them based on sigma_est.
#                 sigma_max = np.max(sigma_est)
#                 if sigma_max < 1e-12:
#                     sigma_max = 1e-12
#                 k_min = 3  # e.g., sigma_max*10^-3
#                 k_max = 1  # e.g., sigma_max*10^-1
#                 num_candidates = 35
#                 lambda_candidates = sigma_max * np.logspace(-k_min, -k_max, num=num_candidates)

#                 best_lambda = None
#                 best_error = np.inf
#                 # Manually perform grid search using the operator error metric.
#                 for candidate in lambda_candidates:
#                     pred_fourier, _ = _compute_prediction_given_lambda(
#                         n, time_array, time_span, kernel_coeff, target_coeff, noise, candidate, X_fourier
#                     )
#                     diff = pred_fourier - target_coeff
#                     op_error = np.sum(avg_power * np.abs(diff) ** 2)
#                     if op_error < best_error:
#                         best_error = op_error
#                         best_lambda = candidate
#                         best_pred_fourier = pred_fourier
#                 prediction_fourier = best_pred_fourier
#             else:
#                 # Use fixed lambda computed from compute_lambda.
#                 lamb = compute_lambda(const, n, r, b)
#                 prediction_fourier, _ = _compute_prediction_given_lambda(
#                     n, time_array, time_span, kernel_coeff, target_coeff, noise, lamb, X_fourier
#                 )

#             diff = prediction_fourier - target_coeff
#             # Compute operator error: sum_{l} [sigma_est[l] * |diff[l]|^2 / kernel_coeff[l]]
#             op_error = np.sum(avg_power * (np.abs(diff) ** 2))
#             errors[j] = op_error

#         op_error_sampmean[n - 1] = np.mean(errors)
#         op_error_sampstd[n - 1] = np.std(errors)

#     return op_error_sampmean, op_error_sampstd





