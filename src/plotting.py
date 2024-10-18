import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles


 # Define colors
red = np.array([141.0, 45.0, 57.0]) / 255.0
dark = np.array([51.0, 51.0, 51.0]) / 255.0




def plot_error_with_std(num_samples, error_sampmean, error_sampstd):
    """
    Plots the mean error with standard deviation as error bars.
    
    Parameters:
    ----------
    num_samples : int
        Number of samples (for the x-axis).
    error_sampmean : numpy.ndarray
        Mean error for each sample (for the y-axis).
    error_sampstd : numpy.ndarray
        Standard deviation of error for each sample (used as error bars).
    xlabel : str, optional
        Label for the x-axis. Default is 'N'.
    ylabel : str, optional
        Label for the y-axis. Default is the provided LaTeX-style error label.
    title : str, optional
        Title of the plot. Default is None.
    
    Returns:
    -------
    None
    """
    index = np.arange(1, num_samples + 1)
    
    # Update matplotlib settings with tueplots for AISTATS 2023 style
    plt.rcParams.update({"figure.dpi": 350})

    # plt.rcParams.update(bundles.aistats2023())
    with plt.style.context(bundles.aistats2023()):
    
        # Create the plot
        fig, ax = plt.subplots(1, )

        # Plot the mean error
        ax.plot(index, error_sampmean, "o--", markersize=4.3, color=dark, alpha=0.9, linewidth=2.3)
        
        # Plot the error bars (standard deviation)
        ax.errorbar(index, error_sampmean, yerr=error_sampstd, color=dark, fmt='o', alpha=0.3, capsize=2.5, ms=2)
        
        # ax.set_yscale('log')  # Optionally set the y-axis to a logarithmic scale


        # Set axis labels and ticks
        ax.set_xlabel('N')
        ax.set_xticks(index[1::5])
        ax.set_ylabel(r'Error \   $ \parallel \tilde{w} - w^* \parallel_{\mathcal{H}} ^2$')
        
        # Optionally set the plot title
        # ax.set_title(title)
        
        # Display the plot
        plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_approximation(time_array, target_f_coeff, time_grid_points, target):
    """
    Plots the true solution against the approximation.

    Parameters:
    ----------
    time_array : numpy.ndarray
        Array of time points where the solution is evaluated.
    target_f_coeff : numpy.ndarray
        Fourier coefficients of the true solution.
    time_grid_points : int
        Number of grid points (resolution).
    target : numpy.ndarray
        True solution values (for comparison).
    red : tuple or np.ndarray
        Color for the true solution plot.
    dark : tuple or np.ndarray
        Color for the approximation plot.
    
    Returns:
    -------
    None
    """
    # Compute the approximation using the inverse FFT
    w = np.fft.ifft(time_grid_points * target_f_coeff)
    
    with plt.style.context(bundles.aistats2023()):
        # Create the plot
        fig, ax = plt.subplots(1,)
        
        # Plot true solution
        ax.plot(time_array, target, label='True solution', color=red)
        
        # Plot approximation
        ax.plot(time_array, w, label='Approximation', color=dark)
        
        # Add a legend
        ax.legend()
        
        # Show the plot
        plt.show()






