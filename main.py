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

'''Main function for UCI letter and spam datasets. //HA
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
from utils import rmse_loss, pfc

def main (args):
  '''Main function for UCI letter and spam datasets.
  
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
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters)
  
  # Report the RMSE performance
  rmse_numerical = rmse_loss(ori_data_x, imputed_data_x, data_m, data_name)

  # Report the PFC performance 
  pfc_categorical = pfc(ori_data_x, imputed_data_x, data_m, data_name)
  
  print()
  print('Dataset: ' + str(data_name))
  print('RMSE - Numerical Performance: ' + str(np.round(rmse_numerical, 4)))
  print('PFC - Categorical Performance: ' + str(np.round(pfc_categorical, 4)) + '%')
  
  return imputed_data_x, rmse_numerical, pfc_categorical

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','news', 'adult', 'mushroom', 'credit', 'basic_test_coded'],
      default='mushroom',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
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
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse_numerical, pfc_categorical = main(args)
encoder = OneHotEncoder()

# Fit the encoder to the data
encoder.fit(imputed_data)

# Revert the transformation using the inverse_transform method
decoded_categorical = encoder.inverse_transform(imputed_data)

# Combine the decoded categorical columns and the numerical columns
#processed_data = np.hstack((decoded_categorical, imputed_data))

# Print the processed data
print(decoded_categorical)
