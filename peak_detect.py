#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:16:33 2024

@author: ERIC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Load the data from the provided file
data = pd.read_csv('spectra_data.txt', delim_whitespace=True)

# Extract frequency (x) and intensity (y)
x = data['freq'].values
y = data['intensity'].values

# Detect peaks in the intensity data
peaks, _ = find_peaks(y, height=np.max(y) * 0.05)  # Detect peaks with at least 5% of the max intensity

# Plot the detected peaks
plt.figure(figsize=(10, 6))
plt.plot(x, y)
#plt.scatter(x[peaks], y[peaks], color='red', label='Detected Peaks', zorder=5)
plt.title('Simulated Spectrum',fontsize=15)
plt.xlabel('Frequency (MHz)', fontsize=15)
plt.ylabel('Intensity (WHz$^{-1}$)', fontsize=15)
plt.legend()
#plt.savefig("simulation.png",dpi=150)
plt.show()

detected_peaks = np.array(x[peaks])

## Define a Gaussian function with fixed mean
def gaussian(x, amplitude, sigma, mean):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

# Define the sum of multiple Gaussians
def multi_gaussian(x, *params):
    gaussians = np.zeros_like(x)
    for i in range(len(detected_peaks)):
        amplitude = params[2 * i]  # Amplitude for Gaussian i
        sigma = params[2 * i + 1]  # Sigma (width) for Gaussian i
        gaussians += gaussian(x, amplitude, sigma, detected_peaks[i])
    return gaussians

# Initial guesses for amplitudes and sigmas
initial_amplitudes = np.ones(len(detected_peaks)) * max(y) / len(detected_peaks)
initial_sigmas = np.ones(len(detected_peaks)) * (x[-1] - x[0]) / (10 * len(detected_peaks))
initial_params = np.column_stack((initial_amplitudes, initial_sigmas)).flatten()

# Fit the multi-Gaussian model to the data
popt, pcov = curve_fit(multi_gaussian, x, y, p0=initial_params)

# Arrays to store the fitted parameters
fitted_means = detected_peaks  # Means are known (detected peaks)
fitted_sigmas = np.zeros(len(detected_peaks))  # Array to store fitted sigmas

# Extract the fitted sigmas (standard deviations)
for i in range(len(detected_peaks)):
    fitted_sigmas[i] = popt[2 * i + 1]  # Extracting sigmas

# Print the mean and standard deviation for each Gaussian
print("Fitted Gaussian Parameters:")
print("Means (detected peaks):", fitted_means)
print("Standard Deviations (sigmas):", fitted_sigmas)

# Plot the original signal and the fitted Gaussian components
plt.figure(figsize=(10, 6))
#plt.scatter(x[peaks], y[peaks], color='red', label='Detected Peaks', zorder=5)


# Plot each Gaussian component
for i in range(len(detected_peaks)):
    amplitude = popt[2 * i]
    sigma = popt[2 * i + 1]
    plt.plot(x, gaussian(x, amplitude, sigma, detected_peaks[i]),
             label=f"Gaussian {i+1} (mean={detected_peaks[i]:.6f})")

# Plot the total fitted signal
fitted_signal = multi_gaussian(x, *popt)
#plt.plot(x, fitted_signal, label="Fitted Signal", color="green", linestyle="--")

plt.title("Signal with Fitted Gaussian Components")
plt.xlabel("Frequency (x)")
plt.ylabel("Intensity (y)")
plt.legend()
plt.show()


