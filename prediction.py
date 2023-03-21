import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from datasets import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

data_name = "credit"
miss_rate = 10

filename_imputed_data = 'imputed_data/{}_{}_wo_target.csv'.format(data_name, miss_rate)
imputed_data_wo_target = pd.read_csv(filename_imputed_data)

filename_original_data = 'preprocessed_data/one_hot_test_data/one_hot_{}_test.csv'.format(data_name)
original_data = pd.read_csv(filename_original_data)

# Split the data into features (X) and target (y)
X = imputed_data_wo_target
y = original_data[datasets[data_name]["target"]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Find best k value
k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))

fig, ax = plt.subplots()

# Plot the data
ax.plot(k_values, scores)

# Set the axis labels and title
ax.set_xlabel("K Values")
ax.set_ylabel("Accuracy Score")
plt.show()

best_index = np.argmax(scores)
best_k = k_values[best_index]

# Create classifier
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate binary case
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("AUROC:", auroc)

# Evaluate multilabel case
