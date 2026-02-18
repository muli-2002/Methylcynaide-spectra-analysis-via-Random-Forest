#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:12:22 2024

@author: ERIC
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def categorize_values(values, bins):
    return np.digitize(values, bins, right=True) - 1
def create_label(array,minimum,maximum,bin_size):
    bins = np.arange(minimum,maximum,bin_size)
    labels = np.arange(len(bins))
    classified_labels = categorize_values(array, bins)
    return classified_labels,labels
def create_logarithmic_label(array, min_exp, max_exp,base=10,step=1):
    """
    Creates logarithmic bins and classifies values into those bins.

    Parameters:
        array (array-like): Input values.
        min_exp (int): Minimum exponent (log scale).
        max_exp (int): Maximum exponent (log scale).
        base (int): Logarithmic base (default: 10).

    Returns:
        array: Classified labels for each value.
    """
    # Use np.logspace to ensure strictly increasing bins
    #bins = np.logspace(min_exp, max_exp, num=int((max_exp - min_exp+1)/step, base=base, dtype=np.float64))
    bins = np.logspace(min_exp, max_exp, num=int((max_exp - min_exp) / step)+1, base=base, dtype=np.float64)
    # Ensure strict monotonicity by removing duplicates (shouldn't happen with logspace)
    bins = np.unique(bins)  
    print(bins)
    classified_labels = categorize_values(array, bins)  # Assign values to bins
    labels = np.arange(len(bins))
    return classified_labels,labels
def compare (test,pred):
    counter = 0
    for i in range(len(test)):
        if test[i] == pred[i]:
            counter+=1
    return counter/len(test)

vector_file = 'vector_3000.txt'
vector_file_2 = 'vector_3000_2.txt'
param_file = 'parameters.txt'
param_file_2 = 'parameters_2000.txt'
error_file = 'corrupt_index.txt'
error_file_2 = 'corrupt_index_2.txt'

vector_file = 'vector_10000.txt'
param_file = 'parameters_10000.txt'
error_file='corrupt_index_10000.txt'

vector = np.genfromtxt(vector_file, delimiter=',',skip_header=0)
param = np.genfromtxt(param_file, delimiter=',',skip_header=0)
error_indices = np.genfromtxt(error_file,skip_header=0, dtype=int)

#vector_2 = np.genfromtxt(vector_file_2, delimiter=',',skip_header=0)
#param_2 = np.genfromtxt(param_file_2, delimiter=',',skip_header=0)
#param_2 = param_2[:2222]
#error_indices_2 = np.genfromtxt(error_file_2,skip_header=0, dtype=int)

# Filter valid error indices (within bounds)
error_indices = error_indices[error_indices < param.shape[0]]
#error_indices_2 = error_indices_2[error_indices_2 < param_2.shape[0]]

param = np.delete(param, error_indices, axis=0)
#param_2 = np.delete(param_2, error_indices_2, axis=0)

filtered_param = param
#filtered_param = np.vstack((param,param_2))
#vector = np.vstack((vector,vector_2))

size = filtered_param[:,0]
temperature = filtered_param[:,1]
density = filtered_param[:,2]

size_density = []

for i in range(len(size)):
    k = (size[i])**2*density[i]*np.pi
    size_density.append(k)


sp_class,sp_label = create_logarithmic_label(size_density,min_exp=12,max_exp=18,base=10)

X_train, X_test, sp_train, sp_test = train_test_split(vector, sp_class, test_size=0.2, random_state=42)

clf_sp = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42,\
                             max_features=None)
clf_sp.fit(X_train, sp_train)
sp_pred = clf_sp.predict(X_test)
accuracy_sp = compare(sp_test,sp_pred)

print(accuracy_sp)

cm = confusion_matrix(sp_test, sp_pred, labels=sp_label)
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.where(row_sums == 0, 0, cm.astype(float) / row_sums)

disp_sp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=sp_label)
disp_sp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on size * column density")
plt.show()

plt.figure(figsize=(8, 5))
bar_width = 0.4

counts_train = [np.sum(sp_train == val) for val in sp_label]
counts_test = [np.sum(sp_test == val) for val in sp_label]

# Plot bars for both arrays
plt.bar(sp_label - bar_width / 2, counts_train, width=bar_width, label="Train", color="blue", alpha=0.7)
plt.bar(sp_label + bar_width / 2, counts_test, width=bar_width, label="Test", color="red", alpha=0.7)

# Labels and title
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Frequency Distribution of combined size density")
plt.xticks(sp_label)  # Ensure x-axis only shows numbers 0 to 5
plt.legend()

# Show the plot
plt.show()

ratio = []
for i in range(len(counts_train)):
    r = counts_test[i]/counts_train[i]
    ratio.append(r)

plt.plot(sp_label,ratio,".")
plt.show()










