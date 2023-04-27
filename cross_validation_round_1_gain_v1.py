from ast import Dict
from itertools import product
import os
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split

import numpy as np
import pandas as pd
import time as td
from data_loader import data_loader
from datasets import datasets

from gain_v1 import gain_v1
from gain_v2 import gain_v2
from utils import rmse_num_loss, rmse_cat_loss, m_rmse_loss
from prediction import linearRegression, kNeighborsClassifier

'''
Description: 

Cross-validation to find optimal parameters per dataset and miss_rate and extra amount. 
Optimization is done based on prediction, and depending on prediction model. 

'''

def main(all_datasets, all_missingness, all_extra_amount):
    df_all_results = pd.DataFrame(columns=['Dataset', 'Missing%', 'Additional CTGAN data%', 'Batch-size',
                    'Hint-rate', 'Alpha', 'AUROC', 'MSE', 'Execution time (s)'])
   
    # Loop through all data sets, miss ratios and extra CTGAN amount
    for dataset in all_datasets:
        for miss_rate in all_missingness:
          for extra_amount in all_extra_amount:
              # Check if the file exists
              if extra_amount != 0:
                file_name = 'preprocessed_data/one_hot_train_data_wo_target_extra_{}/one_hot_{}_train_{}_extra_{}.csv'.format(extra_amount, dataset, miss_rate, extra_amount)
                if not os.path.isfile(file_name):
                  continue
              
              best_params, best_params_score, ex_time = cross_validation_GAIN(dataset, miss_rate, extra_amount)
              if dataset == "news":
                results = {'Dataset': dataset, 'Missing%': miss_rate, 'Additional CTGAN data%': extra_amount, 'Batch-size': best_params['batch_size'],
                      'Hint-rate': best_params['hint_rate'], 'Alpha': best_params['alpha'], 'AUROC': "-", 'MSE': best_params_score, 'Execution time (s)': ex_time}
              else: 
                results = {'Dataset': dataset, 'Missing%': miss_rate, 'Additional CTGAN data%': extra_amount, 'Batch-size': best_params['batch_size'],
                      'Hint-rate': best_params['hint_rate'], 'Alpha': best_params['alpha'],'AUROC': best_params_score, 'MSE': "-", 'Execution time (s)': ex_time}
                
              df_results = pd.DataFrame([results], columns=['Dataset', 'Missing%', 'Additional CTGAN data%', 'Batch-size',
                    'Hint-rate', 'Alpha', 'AUROC', 'MSE', 'Execution time (s)'])
              df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
    
    filename = 'results/optimal_hyperparameters_GAIN_gain_v1_bank.csv'
    df_all_results.to_csv(filename, index=False)


def cross_validation_GAIN(data_name, miss_rate, extra_amount):
    
    print(f'Dataset: {data_name}, Miss rate: {miss_rate}, Extra amount: {extra_amount}')
    start_time = td.time()

    # Load training data and test data
    train_ori_data_x, train_miss_data_x, train_data_m, \
    _, _, _, norm_params_imputation, norm_params_evaluation, _ = data_loader(data_name, miss_rate, extra_amount) 

    # Define the range of hyperparameters to search over
    param_grid = {'batch_size': [64, 128, 256],
                  'hint_rate': [0.1, 0.5, 0.9],
                  'alpha': [0.5, 1, 2, 10],
                  'iterations': [3000]}
    param_combinations = product(*param_grid.values())

    # Define number of cross-folds
    n_folds = 3

    # Create a k-fold cross-validation object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []

    # Loop over all combinations and fit the estimator
    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        all_scores = [] # auroc for all except mse for news

        for train_index, val_index in kf.split(train_miss_data_x):
          # Split in train and validation for fold indexes
          train_x, val_x = train_miss_data_x[train_index], train_miss_data_x[val_index]

          # Perform gain imputation
          imputed_data_val, _ = gain_v1(train_x, val_x, param_dict, data_name, norm_params_imputation)

          # Load data with target
          filename_original_data = 'preprocessed_data/one_hot_train_data/one_hot_{}_train.csv'.format(data_name)
          original_data = pd.read_csv(filename_original_data)

          X = imputed_data_val
          y = original_data[datasets[data_name]["target"]][val_index]

          # Split the data into training and test sets
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
          
          if datasets[data_name]['classification']['model'] == KNeighborsClassifier:
              accuracy, auroc = kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test, data_name)
              score = auroc
          elif datasets[data_name]['classification']['model'] == LinearRegression:
              mse = linearRegression(X_train, X_test, y_train, y_test)
              score = mse
             
          all_scores.append(score)

        # Calculate the mean MSE across all folds for this param combination
        average_score = np.mean(all_scores)
        
        # Add mean to params dict
        if datasets[data_name]['classification']['model'] == KNeighborsClassifier:
          results.append({'params': param_dict, 'AUROC': average_score})
        elif datasets[data_name]['classification']['model'] == LinearRegression:
          results.append({'params': param_dict, 'MSE': average_score})

    # Print all hyperparameters and their corresponding performance metric for this data set
    if datasets[data_name]['classification']['model'] == KNeighborsClassifier:
      for item in results:
        print('Params:', item['params'])
        print('AUROC:', item['AUROC'])

      # Select the parameters  with the highest AUROC score
      best_params = max(results, key=lambda x: x['AUROC'])['params']
      matching_result = next((result for result in results if result['params'] == best_params), None)
      best_params_score = matching_result['AUROC']
      
    elif datasets[data_name]['classification']['model'] == LinearRegression:
      for item in results:
        print('Params:', item['params'])
        print('MSE:', item['MSE'])

      # Select the parameters  with the lowest MSE score
      best_params = min(results, key=lambda x: x['MSE'])['params']
      matching_result = next((result for result in results if result['params'] == best_params), None)
      best_params_score = matching_result['MSE']
    
    end_time = td.time()
    ex_time = end_time - start_time
    ex_time_hours = ex_time / (60*60)
    print(f'Execution time (hours): {ex_time_hours}')

    return best_params, best_params_score, ex_time

if __name__ == '__main__':  

    # Set dataset and missrate
    #all_datasets = ["mushroom", "letter", "bank", "credit", "news"]
    #all_missingness = [10, 30, 50]
    #all_extra_amounts = [0, 50, 100]

    all_datasets = ["bank"]
    all_missingness = [10, 30, 50]
    all_extra_amounts = [0]

    main(all_datasets, all_missingness, all_extra_amounts)



'''from ast import Dict
from itertools import product
import numpy as np
import pandas as pd
from data_loader import data_loader
from datasets import datasets

from sklearn.model_selection import KFold
from gain import gain
from utils import rmse_num_loss, rmse_cat_loss, m_rmse_loss


OLD VERSION
def main(data_name, miss_rate, extra_amount):
  # Load training data and test data
  train_ori_data_x, train_miss_data_x, train_data_m, \
  _, _, _, norm_params_imputation, norm_params_evaluation, _ = data_loader(data_name, miss_rate, extra_amount) 

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
        imputed_data_val, MSE_final = gain(train_x, val_x, param_dict, data_name, norm_params_imputation)  

        # Evaluate performance
        rmse_num = rmse_num_loss(val_full, imputed_data_val, val_m, data_name, norm_params_evaluation)
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
    data_name = "bank"
    miss_rate = 10
    extra_amount = 100

    main(data_name, miss_rate, extra_amount)'''