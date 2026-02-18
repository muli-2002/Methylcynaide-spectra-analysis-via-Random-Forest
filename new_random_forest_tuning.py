#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:12:22 2024

@author: ERIC
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
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
def create_logarithmic_label(array, min_exp, max_exp,base=10,step=0.5):
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
    #print(bins)
    classified_labels = categorize_values(array, bins)  # Assign values to bins
    labels = np.arange(len(bins))
    return classified_labels,labels
def compare (test,pred):
    counter = 0
    for i in range(len(test)):
        if test[i] == pred[i]:
            counter+=1
    return counter/len(test)

localPath_input = './input/'
localPath_vector = './vector/vector_v6/'

vector_file = localPath_vector + 'vector_v6_no_noise.txt'

#vector_file = 'vector_100000_100.txt'

param_file = localPath_input + 'parameters_10000.txt'


vector = np.genfromtxt(vector_file, delimiter=',',skip_header=0)
param_0 = np.genfromtxt(param_file, delimiter=',',skip_header=0)
#param_0 = np.tile(param_0, (10, 1)) #repeat the vector array for 10 times
#error_indices = np.genfromtxt(error_file,skip_header=0, dtype=int)

# Filter valid error indices (within bounds)
#error_indices = error_indices[error_indices < param_0.shape[0]]

#param = np.delete(param_0, error_indices, axis=0)
param = param_0

filtered_param = param
#filtered_param = np.vstack((param,param_2))
#vector = np.vstack((vector,vector_2))

density_index = []
for i in range(len(param_0)):
    if param_0[:,2][i]>1e+16:
        density_index.append(i)


size = filtered_param[:,0]
temperature = filtered_param[:,1]
density = filtered_param[:,2]

size_density = []

for i in range(len(size)):
    k = (size[i])**2*density[i]*np.pi
    size_density.append(k)

density_class,density_label = create_logarithmic_label(density,min_exp=14,max_exp=16.5,base=10)
t_class,t_label = create_label(temperature,50,550,50)
size_class,size_label = create_label(size,0,2.0,0.5)
sp_class,sp_label = create_logarithmic_label(size_density,min_exp=12,max_exp=18,base=10)


# Train-test split
X_train, X_test, density_train, density_test, t_train, t_test, size_train,size_test,\
    sp_train,sp_test\
    = train_test_split(vector, density_class, t_class, size_class, sp_class,\
                       test_size=0.2, random_state=42)

plt.figure(figsize=(8, 5))
bar_width = 0.4

counts_train = [np.sum(density_train == val) for val in density_label]
counts_test = [np.sum(density_test == val) for val in density_label]

# Plot bars for both arrays
plt.bar(density_label - bar_width / 2, counts_train, width=bar_width, label="Train", color="blue", alpha=0.7)
plt.bar(density_label + bar_width / 2, counts_test, width=bar_width, label="Test", color="red", alpha=0.7)

# Labels and title
plt.xlabel("Class")
plt.ylabel("Count")
#plt.title("Distribution of log(Column Density)")
plt.xticks(density_label)  # Ensure x-axis only shows numbers 0 to 5
plt.legend()

# Show the plot
plt.show()

plt.figure(figsize=(8, 5))
bar_width = 0.4

counts_train = [np.sum(t_train == val) for val in t_label]
counts_test = [np.sum(t_test == val) for val in t_label]

# Plot bars for both arrays
plt.bar(t_label - bar_width / 2, counts_train, width=bar_width, label="Train", color="blue", alpha=0.7)
plt.bar(t_label + bar_width / 2, counts_test, width=bar_width, label="Test", color="red", alpha=0.7)

# Labels and title
plt.xlabel("Class")
plt.ylabel("Count")
#plt.title("Distribution of Temperature")
plt.xticks(t_label)  # Ensure x-axis only shows numbers 0 to 5
plt.legend()

# Show the plot
plt.show()

plt.figure(figsize=(8, 5))
bar_width = 0.4

counts_train = [np.sum(t_train == val) for val in size_label]
counts_test = [np.sum(t_test == val) for val in size_label]

# Plot bars for both arrays
plt.bar(size_label - bar_width / 2, counts_train, width=bar_width, label="Train", color="blue", alpha=0.7)
plt.bar(size_label + bar_width / 2, counts_test, width=bar_width, label="Test", color="red", alpha=0.7)

# Labels and title
plt.xlabel("Class")
plt.ylabel("Count")
#plt.title("Distribution of Size")
plt.xticks(size_label)  # Ensure x-axis only shows numbers 0 to 5
plt.legend()

# Show the plot
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
#plt.title("Distribution of log(Size^2 * Density)")
plt.xticks(sp_label)  # Ensure x-axis only shows numbers 0 to 5
plt.legend()

# Show the plot
plt.show()


# Train Random Forest Classifier

from sklearn.model_selection import GridSearchCV

def initialise(X_train, y_train, target_name, output_dir='models', \
                                         save_model=False):
    """
    Train a Random Forest Classifier with Grid Search hyperparameter tuning.
    
    Parameters:
        X_train (array): Training features.
        y_train (array): Training labels.
        target_name (str): Name of the target variable (e.g., 'density').
        output_dir (str): Directory to save the trained model.
        save_model (bool): Whether to save the model to disk.
    
    Returns:
        best_model: Trained Random Forest model with best parameters.
        best_params: Dictionary of best hyperparameters.
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(criterion='entropy', random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    # Train the model
    print(f'Training {target_name} with GridSearch...')
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters for {target_name}: {best_params}")
    
    # Save the model
    if save_model:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(best_model, os.path.join(output_dir, f'{target_name}_classifier_tuned.pkl'))
    
    return best_model, best_params

#clf_density, clf_density_param = initialise(X_train, density_train, 'density', output_dir='models', \
         #                                save_model=False)
clf_temperature, clf_temperature_param = initialise(X_train, t_train, 'temperature', output_dir='models', \
                                         save_model=False)
#clf_size, clf_size_param = initialise(X_train, size_train, 'size', output_dir='models', \
                #                         save_model=False)
#clf_sp, clf_sp_param = initialise(X_train, sp_train, 'sp', output_dir='models', \
                 #                        save_model=False)

# Predict and evaluate
#density_pred = clf_density.predict(X_test)
t_pred = clf_temperature.predict(X_test)
#size_pred = clf_size.predict(X_test)
#sp_pred = clf_sp.predict(X_test)


#accuracy_density = compare(density_test,density_pred)
accuracy_temperature = compare(t_test,t_pred)
#accuracy_size = compare(size_test,size_pred)
#accuracy_sp = compare(sp_test,sp_pred)

#print(accuracy_density)
print(accuracy_temperature)
#print(accuracy_size)
#print(accuracy_sp)


cm = confusion_matrix(density_test, density_pred, labels=density_label)
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.where(row_sums == 0, 0, cm.astype(float) / row_sums)

#cm_formatted = np.where(cm_normalized < 0.01, ["{:.1e}".format(v) for v in cm_normalized.flatten()], cm_normalized)
#cm_formatted = cm_formatted.reshape(cm_normalized.shape)

disp_density = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=density_label)
disp_density.plot(cmap=plt.cm.Blues)
#plt.title("Confusion Matrix on column density")
plt.show()

cm = confusion_matrix(t_test, t_pred, labels=t_label)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
disp_t = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=t_label)
disp_t.plot(cmap=plt.cm.Blues, values_format="1.2g")
#plt.title("Confusion Matrix on temperature")
plt.show()

cm = confusion_matrix(size_test, size_pred, labels=size_label)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
disp_size = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=size_label)
disp_size.plot(cmap=plt.cm.Blues)
#plt.title("Confusion Matrix on size")
plt.show()

cm = confusion_matrix(sp_test, sp_pred, labels=sp_label)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
disp_size = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=sp_label)
disp_size.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on size_density")
plt.show()

###saving models###

output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

#joblib.dump(clf_density, os.path.join(output_dir,'density_classifier100.pkl'))
#joblib.dump(clf_temperature, os.path.join(output_dir,'temperature_classifier100.pkl'))
#joblib.dump(clf_size, os.path.join(output_dir,'size_classifier100.pkl'))
#joblib.dump(clf_sp, os.path.join(output_dir,'sp_classifier100.pkl'))













