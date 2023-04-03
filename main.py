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
from utils import rmse_num_loss, rmse_cat_loss, pfc, m_rmse_loss
from datasets import datasets

def main (args):
  '''Main function for GAIN algorithm.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  extra_amount = args.extra_amount
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load training data and test data
  train_ori_data, train_miss_data, train_data_m, \
  test_ori_data, test_miss_data, test_data_m, norm_params_imputation, norm_params_evaluation, column_names = data_loader(data_name, miss_rate, extra_amount) 
  
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
  pfc_categorical = pfc(test_ori_data, test_imputed_data, test_data_m, data_name)
  
  if rmse_num != None:
    rmse_num = np.round(rmse_num, 4)
  if rmse_cat != None:
    rmse_cat = np.round(rmse_cat, 4)
  if pfc_categorical != None:
    pfc_categorical = np.round(pfc_categorical, 4)
  np.round(m_rmse, 4)

  # Print the results
  print()
  print('Dataset: ' + str(data_name))
  print('mRMSE: ' + str(m_rmse))
  print('RMSE - Numerical Performance: ' + str(rmse_num))
  print('RMSE - Categorical Performance: ' + str(rmse_cat))
  print('PFC - Categorical Performance: ' + str(pfc_categorical))
  print('Execution time: ' + str(ex_time))

  # Save imputed data to csv
  if extra_amount == 0:
    filename_imputed = 'imputed_data/{}_{}_wo_target.csv'.format(data_name, miss_rate)
  else:
    filename_imputed = 'imputed_data/{}_{}_wo_target_extra_{}.csv'.format(data_name, miss_rate, extra_amount)

  df = pd.DataFrame(test_imputed_data, columns=column_names)
  df.to_csv(filename_imputed, index=False)
  
  return test_imputed_data, m_rmse, rmse_num, rmse_cat, pfc_categorical

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
      default=10,
      type=int)
  
  args = parser.parse_args() 

  ## Modify optimal GAIN parameters
  args.data_name = "bank"
  args.miss_rate = 10
  args.extra_amount = 0
  args.iterations = 10000

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
  imputed_data, m_rmse, rmse_num, rmse_cat, pfc_categorical = main(args)



