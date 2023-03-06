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
from sklearn.preprocessing import OneHotEncoder

from data_loader import data_loader
from gain import gain
from utils import rmse_num_loss, rmse_cat_loss, pfc

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
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load training data and test data
  train_ori_data_x, train_miss_data_x, train_data_m, \
  test_ori_data_x, test_miss_data_x, test_data_m = data_loader(data_name, miss_rate)  
  
  # Impute missing data for test data
  test_imputed_data_x = gain(train_miss_data_x, test_miss_data_x, gain_parameters)
  
  # Report the numerical RMSE performance for test data
  rmse_num = rmse_num_loss(test_ori_data_x, test_imputed_data_x, test_data_m, data_name)

  # Report the categorical RMSE performance for test data
  rmse_cat = rmse_cat_loss(test_ori_data_x, test_imputed_data_x, test_data_m, data_name)

  # Report the PFC performance for test data
  pfc_categorical = pfc(test_ori_data_x, test_imputed_data_x, test_data_m, data_name)
  
  print()
  print('Dataset: ' + str(data_name))
  print('RMSE - Numerical Performance: ' + str(np.round(rmse_num, 4)))
  print('RMSE - Categorical Performance: ' + str(np.round(rmse_cat, 4)))
  print('PFC - Categorical Performance: ' + str(np.round(pfc_categorical, 4)) + '%')
  
  return test_imputed_data_x, rmse_num, rmse_cat, pfc_categorical

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','news', 'bank', 'mushroom', 'credit', 'basic_test_coded'],
      default='bank',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data percentage',
      default=10,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=1000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse_num, rmse_cat, pfc_categorical = main(args)

