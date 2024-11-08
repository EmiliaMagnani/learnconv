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


def plot_compare_error_with_std(num_samples, error_sampmean1, error_sampmean2, error_sampstd1, error_sampstd2, xlabel="N", ylabel=r'Error $\parallel \tilde{w} - w^* \parallel_{\mathcal{H}} ^2$', title=None):
    """
    Plots two mean errors with standard deviations as error bars for comparison.
    
    Parameters:
    ----------
    num_samples : int
        Number of samples (for the x-axis).
    error_sampmean1 : numpy.ndarray
        Mean error for the first sample (for the y-axis).
    error_sampmean2 : numpy.ndarray
        Mean error for the second sample (for the y-axis).
    error_sampstd1 : numpy.ndarray
        Standard deviation of error for the first sample (used as error bars).
    error_sampstd2 : numpy.ndarray
        Standard deviation of error for the second sample (used as error bars).
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
    
    # Update matplotlib settings
    plt.rcParams.update({"figure.dpi": 350})

    # use neurips2024 style
    with plt.style.context(bundles.neurips2024()):
    
        fig, ax = plt.subplots()

        # Plot the first mean error with error bars
        ax.errorbar(index, error_sampmean1, yerr=error_sampstd1, fmt='o--', markersize=4.3, color="blue", alpha=0.9, linewidth=2.3, label="Error 1", capsize=2.5)
        
        # Plot the second mean error with error bars
        ax.errorbar(index, error_sampmean2, yerr=error_sampstd2, fmt='s--', markersize=4.3, color="green", alpha=0.9, linewidth=2.3, label="Error 2", capsize=2.5)
        
        # Set axis labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(index[::max(1, num_samples // 10)])  # Adjust x-ticks for readability
        ax.legend()
        
        if title:
            ax.set_title(title)
        
        # Display the plot
        plt.show()


def plot_true_vs_approximation(time_array, target, prediction):
    """
    Plots the true solution and the approximation.

    Parameters:
    ----------
    time_array : numpy.ndarray
        Array of time points.
    prediction : numpy.ndarray
        Approximation of the true solution.
    target : numpy.ndarray
        True solution.
    
    Returns:
    -------
    None
    """
    
    
    # Create the plot
    fig, ax = plt.subplots(1,)
    
    # Plot true solution
    ax.plot(time_array, target, label='True solution', color=red)
    
    # Plot approximation
    ax.plot(time_array, prediction, label='Approximation', color=dark)
    
    # Add a legend
    ax.legend()
    
    # Show the plot
    plt.show()






