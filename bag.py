#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:04:38 2025

@author: ERIC
"""

import numpy as np

# Generate a sample dataset
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)  # Normally distributed data

# Direct Method: Compute the mean of the dataset directly
direct_mean = np.mean(data)

# Bagging Procedure: Compute the mean using bootstrapping
num_bootstrap_samples = 1000
bootstrap_means = []
for _ in range(num_bootstrap_samples):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

# Bagging mean (mean of bootstrap sample means)
bagging_mean = np.mean(bootstrap_means)

# Print results
print("Direct Method Mean:", direct_mean)
print("Bagging Procedure Mean:", bagging_mean)

# Optional: Plot the distribution of bootstrap means for visualization
import matplotlib.pyplot as plt
plt.hist(bootstrap_means, bins=30, alpha=0.7, label='Bootstrap Means')
plt.axvline(direct_mean, color='red', linestyle='dashed', linewidth=1, label='Direct Mean')
plt.axvline(bagging_mean, color='blue', linestyle='dashed', linewidth=1, label='Bagging Mean')
plt.title('Distribution of Bootstrap Means')
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()