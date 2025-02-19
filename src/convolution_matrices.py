import numpy as np
from scipy.linalg import toeplitz

def toeplix_matrix(x):
    """
    Constructs a convolution matrix (using a Toeplitz structure) for a 1D array x,
    returning the central n x n block so that the operation corresponds to the 'same'
    linear convolution of x with an input signal.

    Parameters
    ----------
    x : array_like
        A 1D array representing the convolution kernel.

    Returns
    -------
    ndarray
        A 2D matrix of shape (n, n) that performs convolution with x,
        where n = len(x).
    """
    x = np.asarray(x)
    n = x.size
    
    # Zero pad with n-1 zeros.
    zeros_added = np.zeros(n - 1, dtype=x.dtype)
    
    # Build the full column and row vectors for the Toeplitz matrix.
    column_vector = np.concatenate((x, zeros_added))
    row_vector = np.concatenate((x[:1], zeros_added))
    
    # Construct the full Toeplitz matrix.
    full_matrix = toeplitz(column_vector, row_vector)
    
    # The full matrix has shape (2n-1, n). To obtain the "same" convolution,
    # we extract the central n rows. We compute the starting index as:
    start = (full_matrix.shape[0] - n) // 2
    
    return full_matrix[start : start + n, :]



def circulant_convolution_matrix(v):
    """
    Constructs a circulant matrix from a 1D vector v such that
    multiplication by the matrix corresponds to circular convolution.
    
    Specifically, the returned matrix C has entries
        C[i, j] = v[(i - j) mod n]
    so that for any vector x,
        (C x)[i] = sum_j v[(i - j) mod n] * x[j],
    which is one common way to define circular convolution.
    
    Parameters
    ----------
    v : array_like
        A 1D array representing the convolution kernel.
    
    Returns
    -------
    C : ndarray
        A 2D circulant matrix (of shape (n, n)) that implements circular convolution
        with the kernel v.
    """
    v = np.asarray(v)
    n = v.size
    # Initialize an empty matrix of shape (n, n)
    C = np.empty((n, n), dtype=v.dtype)
    
    # Fill in the circulant matrix: each column is a cyclic shift of v.
    # One common convention: let the first column be v, then
    # C[i, j] = v[(i - j) mod n]
    for j in range(n):
        C[:, j] = np.roll(v, j)
    
    return C