import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define test data (actual values) and predictions
array1 = [4, 0, 4, 2, 0, 3, 4, 4, 2, 1, 0, 4, 1, 2, 3, 2, 1, 1, 0, 2, 3, 4,
       4, 3, 1, 3, 3, 1, 1, 0, 3, 0, 1, 4, 0, 0, 4, 3, 4, 2, 3, 2, 0, 1,
       4, 1, 0, 3, 4, 1, 3, 2, 1, 0, 1, 2, 3, 4, 2, 2, 0, 4, 0, 0, 3, 2,
       3, 0, 3, 4, 4, 1, 3, 3, 2, 4, 2, 2, 1, 4, 3, 4, 0, 4, 2, 1, 4, 2,
       4, 3, 3, 1, 2, 4, 4, 3, 3, 1, 3, 4, 2, 3, 0, 0, 2, 4, 0, 1, 2, 4,
       4, 4, 2, 4, 2, 0, 0, 4, 4, 1, 1, 1, 2, 4, 1, 0, 4, 0, 0, 2, 3, 4,
       0, 2, 0, 1, 4, 2, 0, 3, 1, 2, 3, 2, 4, 4, 2, 3, 4, 2, 1, 1, 2, 3,
       0, 3, 0, 3, 4, 2, 3, 1, 2, 0, 3, 4, 4, 2, 1, 1, 3, 4, 1, 4, 2, 2,
       1, 2, 2, 0, 4, 4, 4, 3, 1, 4, 1, 3, 3, 1, 3, 3]

array2 = [3, 1, 4, 3, 0, 3, 4, 4, 2, 1, 0, 4, 0, 1, 3, 3, 1, 1, 0, 2, 3, 4,
       4, 3, 1, 3, 3, 1, 1, 1, 3, 0, 1, 4, 0, 0, 4, 3, 4, 2, 2, 2, 0, 1,
       3, 1, 0, 2, 3, 1, 3, 3, 1, 0, 0, 2, 3, 4, 1, 2, 0, 4, 0, 1, 3, 2,
       3, 0, 3, 4, 4, 1, 2, 3, 2, 4, 2, 2, 1, 3, 2, 4, 0, 4, 2, 1, 4, 2,
       4, 4, 3, 1, 2, 2, 4, 2, 3, 1, 4, 4, 3, 2, 0, 0, 2, 2, 0, 1, 2, 4,
       4, 4, 2, 4, 2, 0, 0, 4, 4, 1, 1, 3, 2, 4, 0, 0, 4, 4, 0, 2, 2, 4,
       1, 2, 1, 0, 4, 1, 0, 4, 1, 2, 3, 1, 3, 3, 2, 3, 4, 2, 1, 1, 2, 2,
       0, 3, 0, 3, 4, 1, 3, 1, 2, 1, 3, 4, 4, 2, 1, 1, 3, 4, 2, 4, 3, 3,
       1, 2, 2, 0, 4, 4, 3, 2, 1, 4, 1, 4, 3, 1, 4, 3]

# Define class labels with ranges
labels = ["50-150", "150-250", "250-350", "350-450", "450-550"]

# Compute confusion matrix
cm = confusion_matrix(array1, array2, labels=[0, 1, 2, 3, 4])

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix with Range Labels")
plt.show()
