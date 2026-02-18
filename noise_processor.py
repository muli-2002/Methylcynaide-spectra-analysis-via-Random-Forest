#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 21:03:17 2025

@author: ERIC
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ----------- USER PARAMETERS -----------
input_folder = "./plot"  # Folder containing .dat files
output_folder = "./noise_data"  # Folder to save processed files

# Peak detection parameters
peak_fraction = 0.15
SNR = 5
noise_fraction = 1 / SNR
# ---------------------------------------

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each .dat file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".dat"):
        file_path = os.path.join(input_folder, filename)
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file)  # Assuming data is stored as a pickle object
        
        freq = data[:, 0]/1e3  # First column: Frequency
        #print(np.shape(freq))
        intensity = data[:, 1]  # Second column: Intensity
        #print(np.max(intensity))
        
        freqMask = (freq > 238.8) & (freq < 240.0)
        freq = freq[freqMask]
        intensity = intensity[freqMask]
        
        # Add noise
        noise = np.random.normal(0, noise_fraction * np.max(intensity), size=intensity.shape)
        intensity_noisy = intensity + noise
        
        # Save the modified data
        output_file_path = os.path.join(output_folder, f"noise_{filename}")
        with open(output_file_path, 'wb') as output_file:
            pickle.dump(np.column_stack((freq, intensity_noisy)), output_file)
        
        print(f"Processed and saved: {output_file_path}")

print("Processing complete.")
