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

'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''
# Import datasets information
import itertools
from datasets import datasets

# Necessary packages
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim

def get_hyperparameters(dataset, miss_rate, extra_amount):
    '''Get optimized hyperparameters per dataset and miss_rate.
  
  Args:
    - dataset: data name
    - extra_amount: extra amount CTGAN data %
    - miss_rate: missing%
  
  Returns:
    - Batch-size: hyperparameter
    - Hint-rate: hyperparameter
    - Alpha: hyperparameter
  '''  
    # Read dataframe
    filename = 'results/optimal_hyperparameters_GAIN_round_1_gain_v1.csv'
    df = pd.read_csv(filename)

    # Filter dataframe
    df_filtered = df.loc[(df['Dataset'] == dataset) & (df['Missing%'] == miss_rate)]

    # Extract values
    batch_size = df_filtered.at[df_filtered.index[0], 'Batch-size']
    hint_rate = df_filtered.at[df_filtered.index[0], 'Hint-rate']
    alpha = df_filtered.at[df_filtered.index[0], 'Alpha']

    return batch_size, hint_rate, alpha


def reconstruction_loss_function_test(data_name, X, G_sample, M, num_cols_mask):
  '''Reconstruction loss function for test.
  
  Args:
    - data_name: data name
    - X: original data
    - G_sample: generated sample
    - M: data mask
    - num_cols_mask = numerical columns mask
  
  Returns:
    - MSE_loss: mean squared error loss
    - CE_loss: cross-entropy loss
  '''  
  MSE_loss = F.mse_loss((1-M) * X * num_cols_mask, (1-M) * G_sample * num_cols_mask, reduction="mean") / torch.mean((1-M) * num_cols_mask)

  if torch.isnan(MSE_loss).any():
    MSE_loss = torch.tensor(0.0, dtype=torch.float)

  masked_X = (1-M) * X * (1 - num_cols_mask)
  masked_G_sample = (1-M) * G_sample * (1 - num_cols_mask)

  ## Loop through each categorical feature and compute CE loss
  if len(datasets[data_name]['cat_cols']) == 0:
    CE_loss = torch.tensor(0.0, dtype=torch.float)

  else: 
    CE_loss = 0
    variable_sizes = datasets[data_name]['cat_cols'].values()
    start = len(datasets[data_name]['num_cols'])
      
    for variable_size in variable_sizes:
        end = start + variable_size
        batch_G_sample = masked_G_sample[:, start:end]
        batch_X = torch.argmax(masked_X[:, start:end], dim=1)
        CE_loss += F.cross_entropy(batch_G_sample, batch_X, reduction="mean")
        start = end
  
  return MSE_loss, CE_loss

def reconstruction_loss_function_train(data_name, New_X, G_sample, M, num_cols_mask):
  '''Reconstruction loss function for train.
  
  Args:
    - data_name: data name
    - New_X: original data combined with random sample
    - G_sample: generated sample
    - M: data mask
    - num_cols_mask = numerical columns mask
  
  Returns:
    - MSE_loss: mean squared error loss
    - CE_loss: cross-entropy loss
  '''  
  MSE_loss = F.mse_loss(M * New_X * num_cols_mask, M * G_sample * num_cols_mask, reduction="mean") / torch.mean(M * num_cols_mask) # same result as original code
  
  if torch.isnan(MSE_loss).any():
    MSE_loss = torch.tensor(0.0, dtype=torch.float)
  
  masked_New_X = M * New_X * (1 - num_cols_mask)
  masked_G_sample = M * G_sample * (1 - num_cols_mask)

  ## Loop through each categorical feature and compute CE loss
  if len(datasets[data_name]['cat_cols']) == 0:
    CE_loss = torch.tensor(0.0, dtype=torch.float)
  else: 
    CE_loss = 0
    variable_sizes = datasets[data_name]['cat_cols'].values()
    start = len(datasets[data_name]['num_cols'])
      
    for variable_size in variable_sizes:
        end = start + variable_size
        batch_G_sample = masked_G_sample[:, start:end]
        batch_New_X = torch.argmax(masked_New_X[:, start:end], dim=1)
        CE_loss += F.cross_entropy(batch_G_sample, batch_New_X, reduction="mean")
        start = end
  
  return MSE_loss, CE_loss

def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''  

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MinMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
  
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] - min_val[i] + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] - min_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] - min_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data

def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data

def rounding_categorical(imputed_data, data_x, data_name):
  '''Round imputed data for categorical variables. 
  Ensure to only get one "1" per categorical feature.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  no, dim = data_x.shape
  rounded_data = imputed_data.copy()

  n_num_cols = len(datasets[data_name]["num_cols"])
  cat_cols = datasets[data_name]["cat_cols"]

  if (len(cat_cols) == 0):
    return rounded_data

  # Add the start indexes for each categorical feature
  cat_cols_start_indexes = cat_cols.copy()
  cumulative_sums = [0] + list(itertools.accumulate(cat_cols_start_indexes.values()))
  start_indexes = [x + n_num_cols for x in cumulative_sums[:-1]]

  for i, feature_name in enumerate(cat_cols_start_indexes.keys()):
    start_index = start_indexes[i]
    cat_cols_start_indexes[feature_name] = start_index
  
  # Loop through each value in the matrix
  row = 0
  while row < no:
      col = n_num_cols
      while col < dim:
          # check if the current value is NaN
          if np.isnan(data_x[row, col]):        
              for feature_name, index_value in cat_cols_start_indexes.items():
                if index_value == col: # We found the correct feature
                  n_categories = cat_cols[feature_name] 
                  break

              # Extract the current value and the next n_categories-1 values
              values = imputed_data[row, col:col+n_categories]
              
              # Find the index of the maximum value
              max_index = np.argmax(values)

              # Set the maximum value to 1 and the rest to 0 in rounded_data
              rounded_data[row, col:col+n_categories] = 0
              rounded_data[row, col+max_index] = 1
              
              # skip the next n_categories values
              col += n_categories - 1
          col += 1
      row += 1
    
  return rounded_data

def rmse_num_loss(ori_data, imputed_data, data_m, data_name, norm_params):
  '''Compute RMSE loss between ori_data and imputed_data for numerical variables
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse_num: Root Mean Squared Error
  '''

  # Find number of numerical columns
  N_num_cols = len(datasets[data_name]["num_cols"])

  if N_num_cols == 0:
    return None
  else: 
    # Normalize with norm_params
    ori_data_norm, _ = normalization(ori_data, norm_params)
    imputed_data_norm, _ = normalization(imputed_data, norm_params) 

    # Extract only the numerical columns
    ori_data_norm_num = ori_data_norm[:, :N_num_cols]
    imputed_data_norm_num = imputed_data_norm[:, :N_num_cols]
    data_m_num = data_m[:, :N_num_cols]
    
    # Calculate RMSE numerical   
    nominator = np.sum(((1-data_m_num) * ori_data_norm_num - (1-data_m_num) * imputed_data_norm_num)**2)
    denominator = np.sum(1-data_m_num)
    
    rmse_num = np.sqrt(nominator/float(denominator))
    
    return rmse_num

def rmse_cat_loss(ori_data, imputed_data, data_m, data_name):
  '''Compute RMSE loss between ori_data and imputed_data for categorical variables
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse_cat: Root Mean Squared Error
  '''

  # Find number of columns
  N_num_cols = len(datasets[data_name]["num_cols"])   # Find number of numerical columns
  N_cat_cols = len(datasets[data_name]["cat_cols"])   # Find number of categorical columns
  
  if N_cat_cols == 0:
    return None
  else:
    # Extract only the categorical columns
    ori_data_cat = ori_data[:, N_num_cols:]
    imputed_data_cat = imputed_data[:, N_num_cols:]
    data_m_cat = data_m[:, N_num_cols:]
    
    # RMSE categorical  
    nominator = np.sum(((1-data_m_cat) * ori_data_cat - (1-data_m_cat) * imputed_data_cat)**2)
    denominator = np.sum(1-data_m_cat)
    
    rmse_cat = np.sqrt(nominator/float(denominator))
    
    return rmse_cat
  
def m_rmse_loss(rmse_num, rmse_cat):
  '''Compute mRMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - m_rmse: modified Root Mean Squared Error
  '''
  if rmse_cat == None: 
    rmse_cat = 0
  if rmse_num == None:
    rmse_num = 0
  
  m_rmse = np.sqrt((rmse_num**2) + (rmse_cat**2))
    
  return m_rmse

def pfc(ori_data, imputed_data, data_m, data_name): # No taking into consideration category belonging now, to be fixed
  '''Compute PFC between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - pfc: Proportion Falsely Classified
  '''
  # Find number of columns
  # Find number of columns
  N_num_cols = len(datasets[data_name]["num_cols"])   # Find number of numerical columns
  N_cat_cols = len(datasets[data_name]["cat_cols"])   # Find number of categorical columns
  
  if N_cat_cols == 0:
    return None
  else: 
    # Extract only the categorical columns
    ori_data_cat = ori_data[:, N_num_cols:]
    imputed_data_cat = imputed_data[:, N_num_cols:]
    data_m_cat = data_m[:, N_num_cols:]

    data_m_bool = ~data_m_cat.astype(bool) # True indicates missing value (=0), False indicates non-missing value (=1)

    N_missing = np.count_nonzero(data_m_cat == 0) # 0 = missing value
    N_correct = np.sum(ori_data_cat[data_m_bool] == imputed_data_cat[data_m_bool])

    # Calculate PFC
    pfc = (1 - (N_correct/N_missing))*100 # Number of incorrect / Number total missing
    
    return pfc
  
def find_average_and_st_dev(values):
  '''Finding the average and standard deviation along a vector of values.
  
  Args:
    - values: vector of values
    
  Returns:
    - average_value
    - st_dev
  '''
  if all(x is None for x in values):
    return None, None
  
  average_value = np.mean(values)
  st_dev = np.std(values)

  return average_value, st_dev

def round_if_not_none(x):
    '''Round if not none
    
    Args:
      - x: value
      
    Returns:
      - rounded_value'''
    if x is not None:
        return round(x, 4)
    return None

def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / np.sqrt(in_dim / 2.)
  return np.random.normal(size = size, scale = xavier_stddev)
      
def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix

def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  

  
############## ORIGINAL RMSE CODE ##############################
'''def rmse_loss (ori_data, imputed_data, data_m):
  Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
    
  # Only for missing values
  nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
  denominator = np.sum(1-data_m)
  
  rmse = np.sqrt(nominator/float(denominator))
  
  return rmse

############## ORIGINAL NORMALIZATION CODE ##############################

  def normalization (data, parameters=None):
  Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data'''