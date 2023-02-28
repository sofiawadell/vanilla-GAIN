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

'''Data loader for UCI letter, spam and MNIST datasets.
'''
# Import datasets
from datasets import datasets

# Necessary packages
import numpy as np
from utils import binary_sampler


def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    train_data_x: original data
    train_miss_data_x: data with missing values
    train_data_m: indicator matrix for missing components

    test_data_x: original data
    test_miss_data_x: data with missing values
    test_data_m: indicator matrix for missing components
  '''
  
  ## Load training data
  if data_name in datasets.keys():
    file_name = 'preprocessed_data/'+data_name+'_train.csv'
    train_data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  else:
    ValueError("Dataset not found")

  # Parameters
  no, dim = train_data_x.shape
  
  # Introduce missing data
  train_data_m = binary_sampler(1-miss_rate, no, dim)
  train_miss_data_x = train_data_x.copy()
  train_miss_data_x[train_data_m == 0] = np.nan

  ## Load test data
  if data_name in datasets.keys():
    file_name = 'preprocessed_data/'+data_name+'_test.csv'
    test_data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  else:
    ValueError("Dataset not found")

  # Parameters
  no, dim = test_data_x.shape
  
  # Introduce missing data
  test_data_m = binary_sampler(1-miss_rate, no, dim)
  test_miss_data_x = test_data_x.copy()
  test_miss_data_x[test_data_m == 0] = np.nan
      
  return train_data_x, train_miss_data_x, train_data_m, test_data_x, test_miss_data_x, test_data_m
