# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets. 
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import time as td
from sklearn.preprocessing import OneHotEncoder

from data_loader import data_loader
from gain import gain
from utils import rmse_num_loss, rmse_cat_loss, pfc, m_rmse_loss, find_average_and_st_dev, round_if_not_none
from datasets import datasets

def main (args):
  '''Main function for GAIN algorithm.
  
  Args:
    - data_name: data name
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    - extra_amount = extra amount CTGAN training data
    - number_of_runs: number of runs to perform
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  extra_amount = args.extra_amount
  no_of_runs = args.number_of_runs
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  results = []
  
  # Load training data and test data
  train_ori_data, train_miss_data, train_data_m, \
  test_ori_data, test_miss_data, test_data_m, norm_params_imputation, norm_params_evaluation, column_names = data_loader(data_name, miss_rate, extra_amount) 
  
  ## part that should be done for no_of_runs
  for i in range(no_of_runs):
    # Start timer
    start_time = td.time()

    # Impute missing data for test data
    test_imputed_data = gain(train_miss_data, test_miss_data, gain_parameters, data_name, norm_params_imputation)

    # End timer
    end_time = td.time()
    ex_time = end_time - start_time
  
    # Report the numerical RMSE performance for test data
    rmse_num = rmse_num_loss(test_ori_data, test_imputed_data, test_data_m, data_name, norm_params_evaluation)

    # Report the categorical RMSE performance for test data
    rmse_cat = rmse_cat_loss(test_ori_data, test_imputed_data, test_data_m, data_name)
  
    # Report the mRMSE performance for test data
    m_rmse = m_rmse_loss(rmse_num, rmse_cat)

    # Report the PFC performance for test data
    pfc_score = pfc(test_ori_data, test_imputed_data, test_data_m, data_name)
  
    results.append({'run number': i, 'data': test_imputed_data, 'scores':{'mRMSE': m_rmse, 'RMSE num': rmse_num, 'RMSE cat': rmse_cat, 'PFC': pfc_score, 'Execution time': ex_time}})

  for i, result in enumerate(results):
    test_imputed_data = result['data']
    if extra_amount == 0:
      filename = 'imputed_data/{}/{}_{}_wo_target_{}.csv'.format(data_name, data_name, miss_rate, i)
    else: 
      filename = 'imputed_data/{}/{}_{}_wo_target_extra_{}_{}.csv'.format(data_name, data_name, miss_rate, extra_amount, i)
    df = pd.DataFrame(test_imputed_data, columns=column_names)
    df.to_csv(filename, index=False)
  
  best_imputed_data = min(results, key=lambda x: x['scores']['mRMSE'])['data']
  average_m_rmse, st_dev_m_rmse = map(round_if_not_none, find_average_and_st_dev([x['scores']['mRMSE'] for x in results]))
  average_rmse_num, st_dev_rmse_num = map(round_if_not_none, find_average_and_st_dev([x['scores']['RMSE num'] for x in results]))
  average_rmse_cat, st_dev_rmse_cat = map(round_if_not_none, find_average_and_st_dev([x['scores']['RMSE cat'] for x in results]))
  average_pfc, st_dev_pfc = map(round_if_not_none, find_average_and_st_dev([x['scores']['PFC'] for x in results]))
  average_exec_time, st_dev_exec_time = map(round_if_not_none, find_average_and_st_dev([x['scores']['Execution time'] for x in results]))
  
  # Print the results
  print()
  print(f"Dataset: {data_name}, Miss_rate: {miss_rate}, Extra CTGAN data amount :{extra_amount}")
  print(f"Average mRMSE: {average_m_rmse}, Standard deviation: {st_dev_m_rmse}")
  print(f"Average RMSE num: {average_rmse_num}, Standard deviation: {st_dev_rmse_num}")
  print(f"Average RMSE cat: {average_rmse_cat}, Standard deviation: {st_dev_rmse_cat}")
  print(f"Average PFC (%): {average_pfc}, Standard deviation: {st_dev_pfc}")
  print(f"Average execution time (sec): {average_exec_time}, Standard deviation: {st_dev_exec_time}")

  # Save imputed data to csv
  #if extra_amount == 0:
    #filename_imputed = 'imputed_data/{}_{}_wo_target.csv'.format(data_name, miss_rate)
  #else:
    #filename_imputed = 'imputed_data/{}_{}_wo_target_extra_{}.csv'.format(data_name, miss_rate, extra_amount)

  #df = pd.DataFrame(best_imputed_data, columns=column_names)
  #df.to_csv(filename_imputed, index=False)
  
  return best_imputed_data, average_m_rmse, st_dev_m_rmse, average_rmse_num, st_dev_rmse_num, average_rmse_cat, st_dev_rmse_cat, average_pfc, st_dev_pfc, average_exec_time, st_dev_exec_time

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','news', 'bank', 'mushroom', 'credit'],
      default='letter',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data percentage',
      default=10,
      type=float)
  parser.add_argument(
      '--extra_amount',
      help='extra amount of training data generated by CTGAN, 50 or 100%',
      default=0,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=256,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.1,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=2,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  parser.add_argument(
      '--number_of_runs',
      help='number of runs',
      default=10,
      type=int)
  
  args = parser.parse_args() 

  all_datasets = ["mushroom", "letter", "bank", "credit", "news"]
  all_miss_rates = [10, 30, 50]
  all_extra_amounts = [0, 50, 100]

  df_results = pd.DataFrame(columns=['Dataset', 'Missing%', 'Additional CTGAN data%', 'Average mRMSE',
                    'St Dev mRMSE', 'Average RMSE num', 'St Dev RMSE num', 'Average RMSE cat', 'St Dev RMSE cat', 
                    'Average PFC (%)', 'St Dev PFC (%)', 'Average execution time (s)', 'St Dev execution time (s)'])
  
  # Start looping through datasets and do imputation
  for dataset in all_datasets:
    for miss_rate in all_miss_rates:
        for extra_amount in all_extra_amounts: 

          if dataset == "news" and miss_rate == 30 and extra_amount !=0:
            continue
          if miss_rate == 50 and extra_amount !=0:
            continue
          
          ## Modify GAIN parameters
          args.data_name = dataset
          args.miss_rate = miss_rate
          args.extra_amount = extra_amount
          args.iterations = 1
          args.number_of_runs = 10

          if args.extra_amount == 0:
            case = "ordinary_case"
          elif args.extra_amount == 50:
            case = "extra_50"
          elif args.extra_amount == 100:
            case = "extra_100"
          else:
            ValueError("Extra amount not chosen correctly, chose 0, 50 or 100")

          args.batch_size = datasets[args.data_name]["optimal_parameters"][case]["batch_size"]
          args.hint_rate = datasets[args.data_name]["optimal_parameters"][case]["hint_rate"]
          args.alpha = datasets[args.data_name]["optimal_parameters"][case]["alpha"]

          # Calls main function  
          best_imputed_data, average_m_rmse, st_dev_m_rmse, average_rmse_num, st_dev_rmse_num, average_rmse_cat, st_dev_rmse_cat, average_pfc, st_dev_pfc, average_exec_time, st_dev_exec_time = main(args)
          results = {'Dataset': dataset, 'Missing%': miss_rate, 'Additional CTGAN data%': extra_amount, 'Average mRMSE': average_m_rmse,
                    'St Dev mRMSE': st_dev_m_rmse, 'Average RMSE num': average_rmse_num, 'St Dev RMSE num': st_dev_rmse_num, 'Average RMSE cat': average_rmse_cat, 'St Dev RMSE cat': st_dev_rmse_cat, 
                    'Average PFC (%)': average_pfc, 'St Dev PFC (%)': st_dev_pfc, 'Average execution time (s)': average_exec_time, 'St Dev execution time (s)': st_dev_exec_time}
          df_results = df_results.append(results, ignore_index=True)
  
  # Save result to csv
  df_results.to_csv('imputed_data/summary.csv', index=False)

'''
ORIGINAL CODE

          ## Modify GAIN parameters
          args.data_name = "letter"
          args.miss_rate = 50
          args.extra_amount = 0
          args.iterations = 10000
          args.number_of_runs = 1

          if args.extra_amount == 0:
            case = "ordinary_case"
          elif args.extra_amount == 50:
            case = "extra_50"
          elif args.extra_amount == 100:
            case = "extra_100"
          else:
            ValueError("Extra amount not chosen correctly, chose 0, 50 or 100")

          args.batch_size = datasets[args.data_name]["optimal_parameters"][case]["batch_size"]
          args.hint_rate = datasets[args.data_name]["optimal_parameters"][case]["hint_rate"]
          args.alpha = datasets[args.data_name]["optimal_parameters"][case]["alpha"]
          
          # Calls main function  
          imputed_data, m_rmse, rmse_num, rmse_cat, pfc, execution_time = main(args)'''

