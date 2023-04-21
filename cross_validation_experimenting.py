from ast import Dict
from itertools import product
import os
import numpy as np
import pandas as pd
from data_loader import data_loader
from datasets import datasets

from sklearn.model_selection import KFold
from gain_v1 import gain
from utils import rmse_num_loss, rmse_cat_loss, m_rmse_loss

'''
Description: 

Cross-validation to find optimal parameters per dataset and miss_rate and extra amount

'''

def main(all_datasets, all_missingness, all_extra_amount):
    df_all_results = pd.DataFrame(columns=['Dataset', 'Missing%', 'Additional CTGAN data%', 'Batch-size',
                    'Hint-rate', 'Alpha', 'Beta', 'Tau', 'MSE', 'CE'])
   
    # Loop through all data sets, miss ratios and extra CTGAN amount
    for dataset in all_datasets:
        for miss_rate in all_missingness:
          for extra_amount in all_extra_amount:
              # Check if the file exists
              if extra_amount != 0:
                file_name = 'preprocessed_data/one_hot_train_data_wo_target_extra_{}/one_hot_{}_train_{}_extra_{}.csv'.format(extra_amount, dataset, miss_rate, extra_amount)
                if not os.path.isfile(file_name):
                  continue
              
              best_params, best_params_mse, best_params_ce = cross_validation_GAIN(dataset, miss_rate, extra_amount)
              results = {'Dataset': dataset, 'Missing%': miss_rate, 'Additional CTGAN data%': extra_amount, 'Batch-size': best_params['batch_size'],
                      'Hint-rate': best_params['hint_rate'], 'Alpha': best_params['alpha'], 'Beta': best_params['beta'], 'Tau': best_params['tau'], 'MSE': best_params_mse, 'CE': best_params_ce}
              
              df_results = pd.DataFrame([results], columns=['Dataset', 'Missing%', 'Additional CTGAN data%', 'Batch-size',
                                                               'Hint-rate', 'Alpha', 'Beta', 'Tau' 'MSE', 'CE'])
              df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
    
    df_all_results.to_csv('results/optimal_hyperparameters_GAIN_round_1.csv', index=False)


def cross_validation_GAIN(data_name, miss_rate, extra_amount):
    print(f'Dataset: {data_name}, Miss rate: {miss_rate}, Extra amount: {extra_amount}')

    # Load training data and test data
    train_ori_data_x, train_miss_data_x, train_data_m, \
    _, _, _, norm_params_imputation, norm_params_evaluation, _ = data_loader(data_name, miss_rate, extra_amount) 
    
    #'batch_size': [64, 128, 256],
                  #'hint_rate': [0.1, 0.5, 0.9],
    # Define the range of hyperparameters to search over
    param_grid = {'batch_size': [256],
                  'hint_rate': [0.5],
                  'alpha': [0.5, 1, 10, 50, 100],
                  'beta': [0.1, 0.5, 1],
                  'tau': [0.1, 0.5, 1, 2, 5],
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
        all_mse = []
        all_ce = []

        for train_index, val_index in kf.split(train_miss_data_x):
          # Split in train and validation for fold indexes
          train_x, val_x = train_miss_data_x[train_index], train_miss_data_x[val_index]

          # Perform gain imputation
          imputed_data_val, MSE_final, CE_final = gain(train_x, val_x, param_dict, data_name, norm_params_imputation)
          all_mse.append(MSE_final.detach().numpy())
          all_ce.append(CE_final.detach().numpy())

        # Calculate the mean MSE across all folds for this param combination
        average_mse = np.mean(all_mse)
        average_ce = np.mean(all_ce)
        
        # Add mean to params dict
        results.append({'params': param_dict, 'MSE': average_mse, 'CE': average_ce})

    # Print all hyperparameters and their corresponding performance metric for this data set
    for item in results:
        print('Params:', item['params'])
        print('MSE:', item['MSE'])
        print('CE:', item['CE'])

    # Select the parameters  with the lowest MSE score
    best_params = min(results, key=lambda x: x['MSE'])['params']
    matching_result = next((result for result in results if result['params'] == best_params), None)
    best_params_mse = matching_result['MSE']
    best_params_ce = matching_result['CE']

    return best_params, best_params_mse, best_params_ce

if __name__ == '__main__':  
    
    # Set dataset and missrate
    all_datasets = ["mushroom", "letter", "bank", "credit", "news"]
    all_missingness = [10, 30, 50]
    all_extra_amounts = [0, 50, 100]

    all_datasets = ["credit"]
    all_missingness = [10]
    all_extra_amounts = [0]

    main(all_datasets, all_missingness, all_extra_amounts)