import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from data_loader import data_loader
from datasets import datasets
from parameter_options import parameter_options

from sklearn.model_selection import KFold
from gain import gain
from utils import rmse_num_loss, rmse_cat_loss, m_rmse_loss

'''
Description: 

Cross-validation round 2 to find optimal parameters per dataset

'''

all_datasets = ["mushroom", "letter", "bank", "credit"]
all_missingness = [10, 30]

def main():
    
    final_parameters = [] 

    for data_name in all_datasets:
        results = []

        # Get all hyperparameter keys and values
        param_keys = list(parameter_options[data_name][1].keys())
        param_values = [list(hp.values()) for hp in parameter_options[data_name].values()]

        # Define number of cross-folds
        n_folds = 5

        # Create a k-fold cross-validation object
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for params in param_values:
            param_dict = dict(zip(param_keys, params))
            all_m_rmse_for_different_miss_rates = []

            # Loop over all missingness ratios 
            for miss_rate in all_missingness:
                all_m_rmse = []

                # Load training data and test data
                train_ori_data_x, train_miss_data_x, train_data_m, \
                _, _, _, norm_params_imputation, norm_params_evaluation, column_names = data_loader(data_name, miss_rate, 100) 

                for train_index, val_index in kf.split(train_miss_data_x):
                    # Split in train and validation for fold indexes
                    train_x, val_x = train_miss_data_x[train_index], train_miss_data_x[val_index]
                    _, val_full = train_ori_data_x[train_index], train_ori_data_x[val_index]
                    _, val_m = train_data_m[train_index], train_data_m[val_index]

                    # Perform gain imputation
                    imputed_data_val = gain(train_x, val_x, param_dict, data_name, norm_params_imputation)  

                    # Evaluate performance
                    rmse_num = rmse_num_loss(val_full, imputed_data_val, val_m, data_name, norm_params_evaluation)
                    rmse_cat = rmse_cat_loss(val_full, imputed_data_val, val_m, data_name)
                    m_rmse = m_rmse_loss(rmse_num, rmse_cat)
        
                    all_m_rmse.append(m_rmse)

                # Calculate the mean RMSE across all folds for this param combination for this missingness ratio
                all_m_rmse_for_different_miss_rates.append(np.mean(all_m_rmse))

            average_m_rmse_per_param_comb = np.mean(all_m_rmse_for_different_miss_rates)
            results.append({'params': params, 'scores':[average_m_rmse_per_param_comb]})

        # Print all hyperparameters and their corresponding performance metric
        print('Dataset:', data_name)
        for item in results:
            print('Params:', item['params'])
            print('Scores:', item['scores'])
        
        # Sort the results list by the first score in each scores list (which is the average_m_rmse_per_param_comb)
        sorted_results = sorted(results, key=lambda x: x['scores'][0])
        best_result = sorted_results[0]
        best_params = best_result['params']

        final_parameters.append({'dataset': data_name, 'best_params':best_params})
    
    return final_parameters

if __name__ == '__main__':  
    final_parameters = main()
    for item in final_parameters:
        print('Dataset:', item['dataset'])
        print('Best parameters:', item['best_params'])

