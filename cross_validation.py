from ast import Dict
from itertools import product
import numpy as np
import pandas as pd
from data_loader import data_loader
from datasets import datasets

from sklearn.model_selection import KFold
from gain import gain
from utils import rmse_num_loss, rmse_cat_loss, m_rmse_loss

'''
Description: 

Cross-validation to find optimal parameters per dataset and miss_rate

'''

def main(data_name, miss_rate):
  # Load training data and test data
  train_ori_data_x, train_miss_data_x, train_data_m, \
  _, _, _, norm_params_train, _ = data_loader(data_name, miss_rate) 

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
      all_m_rmse = []
      all_rmse_num = []
      all_rmse_cat = []

      for train_index, val_index in kf.split(train_miss_data_x):
        # Split in train and validation for fold indexes
        train_x, val_x = train_miss_data_x[train_index], train_miss_data_x[val_index]
        _, val_full = train_ori_data_x[train_index], train_ori_data_x[val_index]
        _, val_m = train_data_m[train_index], train_data_m[val_index]

        # Perform gain imputation
        imputed_data_val = gain(train_x, val_x, param_dict, data_name)  

        # Evaluate performance
        rmse_num = rmse_num_loss(val_full, imputed_data_val, val_m, data_name, norm_params_train)
        rmse_cat = rmse_cat_loss(val_full, imputed_data_val, val_m, data_name)
        m_rmse = m_rmse_loss(rmse_num, rmse_cat)
        
        all_rmse_num.append(rmse_num)
        all_rmse_cat.append(rmse_cat)
        all_m_rmse.append(m_rmse)

        print(f'Hyperparameters: {param_dict}, mRMSE: {m_rmse}, RMSE num: {rmse_num}, RMSE cat: {rmse_cat}')

      # Calculate the mean RMSE across all folds for this param combination
      if all(element == None for element in all_rmse_num):
        average_rmse_num = None
        average_rmse_cat = np.mean(all_rmse_cat)
        average_m_rmse = np.mean(all_m_rmse)
      elif all(element == None for element in all_rmse_cat):
        average_rmse_num = np.mean(all_rmse_num)
        average_rmse_cat = None
        average_m_rmse = np.mean(all_m_rmse)
      else:
        average_rmse_num = np.mean(all_rmse_num)
        average_rmse_cat = np.mean(all_rmse_cat)
        average_m_rmse = np.mean(all_m_rmse)

      # Add mean to params dict
      results.append({'params': param_dict, 'scores':[average_m_rmse, average_rmse_num, average_rmse_cat]})

  # Print all hyperparameters and their corresponding performance metric
  for item in results:
      print('Params:', item['params'])
      print('Scores:', item['scores'])

  # Select the parameters  with the lowest mRMSE score
  best_params = min(results, key=lambda x: x['scores'][0])['params']
  matching_result = next((result for result in results if result['params'] == best_params), None)
  best_params_m_rmse, best_params_rmse_num, best_params_rmse_cat = matching_result['scores']

  print(f'Best parameter selection for mRMSE: {best_params}, mRMSE: {best_params_m_rmse}, RMSE num: {best_params_rmse_num}, RMSE cat: {best_params_rmse_cat}')

if __name__ == '__main__':  
    # Set dataset and missrate
    data_name = "news"
    miss_rate = 50

    main(data_name, miss_rate)