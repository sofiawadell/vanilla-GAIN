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
from datasets import datasets

# Necessary packages
import numpy as np
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()

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
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) - min_val[i] + 1e-6)   
      
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

def rmse_num_loss(ori_data, imputed_data, data_m, data_name):
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
    # Extract only the numerical columns
    ori_data_num = ori_data[:, :N_num_cols]
    imputed_data_num = imputed_data[:, :N_num_cols]
    data_m_num = data_m[:, :N_num_cols]
    
    # RMSE numerical 
    ori_data_num, norm_parameters = normalization(ori_data_num)
    imputed_data_num, _ = normalization(imputed_data_num, norm_parameters)  
    nominator = np.sum(((1-data_m_num) * ori_data_num - (1-data_m_num) * imputed_data_num)**2)
    denominator = np.sum(1-data_m_num)
    
    rmse_num = np.sqrt(nominator/float(denominator))
    
    return rmse_num

######## NEW METHOD ######################
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
    ori_data_cat, norm_parameters = normalization(ori_data_cat)
    imputed_data_cat, _ = normalization(imputed_data_cat, norm_parameters)  
    nominator = np.sum(((1-data_m_cat) * ori_data_cat - (1-data_m_cat) * imputed_data_cat)**2)
    denominator = np.sum(1-data_m_cat)
    
    rmse_cat = np.sqrt(nominator/float(denominator))
    
    return rmse_cat

######## NEW METHOD ######################
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
    pfc = (1 - (N_correct/N_missing))*100
    
    return pfc

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