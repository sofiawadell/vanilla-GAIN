from ast import Dict
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from data_loader import data_loader
from datasets import datasets

from sklearn.model_selection import KFold
from gain import gain
from utils import normalization, rmse_num_loss, pfc

data_name = "news"
miss_rate = 10

 # Load training data and test data
train_ori_data_x, train_miss_data_x, train_data_m, \
test_ori_data_x, test_miss_data_x, test_data_m = data_loader(data_name, miss_rate) 

# Define the range of hyperparameters to search over
param_grid = {'batch_size': [64, 128, 256],
              'hint_rate': [0.1, 0.5, 0.9],
              'alpha': [0.1, 0.5, 1, 2, 10],
              'iterations': [10000]}
param_combinations = product(*param_grid.values())

# Define number of cross-folds
n_folds = 5

# Create a k-fold cross-validation object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

results = []

# Loop over all combinations and fit the estimator
for params in param_combinations:
    param_dict = dict(zip(param_grid.keys(), params))
    all_rmse_num = []
    all_pfc_scores = []

    for train_index, val_index in kf.split(train_miss_data_x):
      # Split in train and validation for fold indexes
      train_x, val_x = train_miss_data_x[train_index], train_miss_data_x[val_index]
      train_full, val_full = train_ori_data_x[train_index], train_ori_data_x[val_index]
      train_m, val_m = train_data_m[train_index], train_data_m[val_index]

      # Perform gain imputation
      imputed_data_val = gain(train_x, val_x, param_dict)  

      # Evaluate performance
      rmse_num = rmse_num_loss(val_full, imputed_data_val, val_m, data_name)
      all_rmse_num.append(rmse_num)

      pfc_score = pfc(val_full, imputed_data_val, val_m, data_name)
      all_pfc_scores.append(pfc_score)

      print(f'Hyperparameters: {param_dict}, RMSE num: {rmse_num}, PFC: {pfc_score}')

    # Calculate the mean RMSE across all folds for this param combination
    average_rmse_num = np.mean(all_rmse_num)
    average_pcf_score = np.mean(all_pfc_scores)

    # Add mean to params dict
    results.append({'params': param_dict, 'scores':[average_rmse_num, average_pcf_score]})

# Print the best hyperparameters and their corresponding performance metric
for item in results:
    print('Params:', item['params'])
    print('Scores:', item['scores'])

# Select the parameters  with the lowest RMSE score
best_params_rmse = min(results, key=lambda x: x['scores'][0])['params']
best_params_pfc = min(results, key=lambda x: x['scores'][1])['params']

print('Best parameter selection for numerical: ', best_params_rmse)
print('Best parameter selection for categorical: ', best_params_pfc)
