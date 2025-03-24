# Learning Convolution Operators with Kernel Methods: Experiments

This repository contains the experimental code associated with the paper:

**Emilia Magnani, Ernesto De Vito, Philipp Hennig, Lorenzo Rosasco.  
*Learning Convolution Operators with Kernel Methods.*  
arXiv:2501.05279 (2025)**

## Repository Structure

- **experiments/**
  - `compute_prediction.py` —  Script to compute predictions using the convolution operators.
  - `error_comparison.py` — Script for comparing for comparing the error metrics for different input signals (frequency localised, space localised)
  - `operator_error.py` —  Script for evaluating operator error.
- **notebooks/**
  - `Heat_equation_pde_application.ipynb` — Notebook demonstrating the heat equation application.
  - `Sobolev_kernel.ipynb` — Explores the Sobolev kernel.
  - `tutorial.ipynb` — A guided walkthrough of experiments and formulas.
- **results/**  
  *(Note: This folder is generated when running experiments and is excluded from version control.)*
- `README.md` — This file.
- `requirements.txt` — Dependencies list.
- `setup.py` — Setup script for the package.
- **src/**
  - `convolution_matrices.py` — Module for generating convolution matrices (circulant and Toeplitz).
  - `fourier_inference.py` — Module for Fourier inference.
  - `fourier.py` — Fourier related functionalities.
  - `generate_input_signals.py` — Generates input signals.
  - `kernels.py` — Kernel definitions and operations.
  - `plotting.py` — Plotting functions.
  - `regularization.py` — Regularization routines.
  - `sampling.py` — Sampling routines.
  - `target_signals.py` — Generates target signals.
- **tests/**
  - `test_fourier.py` — Tests for Fourier functionalities.
  - `test_input_signals_decay.py` — Tests for input signal decay.
  - `test_kernels.py` — Tests for kernel implementations.
  - `test_sampling.py` — Tests for sampling routines.
