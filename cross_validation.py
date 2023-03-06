from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from data_loader import data_loader
from datasets import datasets

from sklearn.model_selection import GridSearchCV, KFold
from gain import gain
from utils import normalization



data_name = "bank"
miss_rate = 10

 # Load training data and test data
train_ori_data_x, train_miss_data_x, train_data_m, \
test_ori_data_x, test_miss_data_x, test_data_m = data_loader(data_name, miss_rate) 

# Define the range of hyperparameters to search over
param_grid = {'batch_size': [1, 2, 3],
              'hint_rate': [0.1, 0.5, 1.0],
              'alpha': [0.1, 0.5, 1.0],
              'iterations': [10, 100, 200]}
param_combinations = product(*param_grid.values())

# Create a k-fold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_num_loss(ori_data, imputed_data, data_m, data_name):
  # Find number of numerical columns
  N_num_cols = len(datasets[data_name]["num_cols"])

  if N_num_cols == 0:
    return None
  else: 
    # Extract only the numerical columns
    ori_data_num = ori_data[:, :N_num_cols]
    imputed_data_num = imputed_data[:, :N_num_cols]
    data_m_num = data_m[:, :N_num_cols]
    
    # RMSE numerical 
    ori_data_num, norm_parameters = normalization(ori_data_num)
    imputed_data_num, _ = normalization(imputed_data_num, norm_parameters)  
    nominator = np.sum(((1-data_m_num) * ori_data_num - (1-data_m_num) * imputed_data_num)**2)
    denominator = np.sum(1-data_m_num)
    
    rmse_num = np.sqrt(nominator/float(denominator))
    
    return rmse_num

# Loop over all combinations and fit the estimator
for params in param_combinations:
    param_dict = dict(zip(param_grid.keys(), params))
    imputed_data_test, imputed_data_train = gain(train_miss_data_x, test_miss_data_x, param_dict)
    score = rmse_num_loss()
    print(f'Hyperparameters: {param_dict}, Score: {score}')

# Print the best hyperparameters and their corresponding performance metric
best_idx = np.argmax(grid_search.cv_results_['mean_test_score'])
best_hyperparams = grid_search.cv_results_['params'][best_idx]
best_score = grid_search.cv_results_['mean_test_score'][best_idx]
print('Best hyperparameters:', best_hyperparams)
print('Best performance metric:', best_score)