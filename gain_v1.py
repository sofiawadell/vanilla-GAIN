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

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import normalization, reconstruction_loss_function_test, renormalization, rounding, rounding_categorical
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index, reconstruction_loss_function_test, reconstruction_loss_function_train
from datasets import datasets

def gain_v1(train_data_x, test_data_x, gain_parameters, data_name, norm_params_imputation):
  '''Impute missing values in test_data_x
  
  Args:
    - train_data_x: original training data with missing values
    - test_data_x: original test data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
    - data_name: name of dataset
    - norm_params_imputation: normalization parameters to be used in imputation
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  train_data_m = 1-np.isnan(train_data_x) # 0 is missing value, 1 is existing value
  test_data_m = 1-np.isnan(test_data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']  

  # Other parameters for training data
  no, dim = train_data_x.shape
  test_no, dim = test_data_x.shape

  # Numerical matrix
  num_cols_len = len(datasets[data_name]['num_cols'])
  num_cols_mask_full = np.zeros((no, dim))
  num_cols_mask_full[:, :num_cols_len] = 1
  num_cols_mask_full = torch.tensor(num_cols_mask_full)

  # Hidden state dimensions
  h_dim = int(dim)
  h_dim = dim

  # Normalization for all columns and test/training
  train_norm_data_x, _ = normalization(train_data_x, norm_params_imputation)
  train_norm_data_x = np.nan_to_num(train_norm_data_x, 0)

  test_norm_data_x, _ = normalization(test_data_x, norm_params_imputation)
  test_norm_data_x = np.nan_to_num(test_norm_data_x, 0)

  ## GAIN architecture   
  # Discriminator variables
  D_W1 = torch.tensor(xavier_init([dim*2, h_dim]), requires_grad=True) # Data + Hint as inputs
  D_b1 = torch.tensor(np.zeros(shape = [h_dim]), requires_grad=True)
  D_W2 = torch.tensor(xavier_init([h_dim, h_dim]), requires_grad=True)
  D_b2 = torch.tensor(np.zeros(shape = [h_dim]), requires_grad=True)
  D_W3 = torch.tensor(xavier_init([h_dim, dim]), requires_grad=True)
  D_b3 = torch.tensor(np.zeros(shape = [dim]), requires_grad=True)  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

  # Generator variables
  G_W1 = torch.tensor(xavier_init([dim*2, h_dim]),requires_grad=True)     
  G_b1 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)
  G_W2 =  torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
  G_b2 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)  
  G_W3 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
  G_b3 = torch.tensor(np.zeros(shape = [dim]),requires_grad=True)

  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(new_x,m):
    inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
    G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
    G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)   
    G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
    
    return G_prob
      
  # Discriminator
  def discriminator(new_x, h):
    # Concatenate Data and Hint
    inputs = torch.cat(dim = 1, tensors = [new_x, h]) 
    D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
    D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
    D_logit = torch.matmul(D_h2, D_W3) + D_b3
    D_prob = torch.sigmoid(D_logit)
    return D_prob
  
  def discriminator_loss(M, New_X, H):
    # Generator
    G_sample = generator(New_X, M)    
    # Combine with observed data    
    Hat_New_X = New_X * M + G_sample * (1-M) # the matrix that only missing values are replaced by G_sample     
    
    # Discriminator         
    D_prob = discriminator(Hat_New_X, H) 

    # Loss
    D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
    return D_loss             
  
  def generator_loss(X, M, New_X, H):
    # Generator
    G_sample = generator(New_X,M)

    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    # Loss
    M = M.float()
    G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
    MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

    G_loss = G_loss1 + alpha * MSE_train_loss 

    # MSE Performance metric
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    return G_loss, MSE_train_loss, MSE_test_loss
  
  def test_loss(X, M, New_X):
    # Generator
    G_sample = generator(New_X,M)

    # MSE Performance metric
    M = M.float()
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    return MSE_test_loss, G_sample

  ## Define optimizers
  D_solver = torch.optim.Adam(params=theta_D)
  G_solver = torch.optim.Adam(params=theta_G)
  
  all_test_loss = []
  all_train_loss = []
  all_d_loss = []
  all_g_loss = []

  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = train_norm_data_x[batch_idx, :]  
    M_mb = train_data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
    # Missing Data Introduce
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  

    X_mb = torch.tensor(X_mb)
    M_mb = torch.tensor(M_mb)
    H_mb = torch.tensor(H_mb)
    New_X_mb = torch.tensor(New_X_mb)

    # Optimize D
    D_solver.zero_grad()
    D_loss = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
    D_loss.backward()
    D_solver.step()
    
    # Optimize G
    G_solver.zero_grad()
    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr, = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
    G_loss_curr.backward()
    G_solver.step() 

    all_train_loss.append(MSE_train_loss_curr)
    all_test_loss.append(MSE_test_loss_curr)

    all_d_loss.append(D_loss)
    all_g_loss.append(G_loss_curr)

    # Intermediate Losses
    if it % 10000 == 0:
        print('Iter: {}'.format(it),end='\t')
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())),end='\t')
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))     
  
  # Plot the losses
  '''all_train_loss_list = [tensor.detach().numpy() for tensor in all_train_loss]
  all_test_loss_list = [tensor.detach().numpy() for tensor in all_test_loss]
  all_d_loss_list = [tensor.detach().numpy() for tensor in all_d_loss]
  all_g_loss_list = [tensor.detach().numpy() for tensor in all_g_loss]
  x = np.arange(len(all_d_loss_list))
  #x = x[:, np.newaxis, np.newaxis].shape

  plt.plot(x, all_d_loss_list, label="D loss")
  plt.plot(x, all_g_loss_list, label="G loss")
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.title('GAIN d and g loss for ' + data_name)
  plt.legend()
  plt.show()'''

  ## Return imputed test data
  Z_mb = uniform_sampler(0, 0.01, test_no, dim) 
  M_mb = test_data_m
  X_mb = test_norm_data_x          
  New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

  X_mb = torch.tensor(X_mb)
  M_mb = torch.tensor(M_mb)
  New_X_mb = torch.tensor(New_X_mb)
      
  MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
  
  imputed_data_test = M_mb * X_mb + (1-M_mb) * Sample

  # Convert to Numpy array
  imputed_data_test = imputed_data_test.detach()
  imputed_data_test = imputed_data_test.numpy()

  # Renormalization for all columns
  imputed_data_test = renormalization(imputed_data_test, norm_params_imputation)  
  
  # Rounding
  imputed_data_test = rounding(imputed_data_test, test_data_x)
          
  return imputed_data_test, MSE_final

'''Generator
  def generator_gumbel(new_x, m, tau=1.0):
    # Concatenate Mask and Data
    inputs = torch.cat(dim=1, tensors=[new_x, m]) 
    G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
    G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)

    # Separate categorical and continuous columns
    con_inputs = G_h2[:, :n_con_cols]
    dis_inputs = G_h2[:, :n_con_cols]

    # Gumbel-Softmax normalized output for categorical columns
    logits_cat = torch.matmul(con_inputs, G_W3[:, :n_con_cols]) + G_b3[:, :n_con_cols]
    G_prob_cat = F.gumbel_softmax(logits_cat, tau=tau, dim=-1)

    # Sigmoid normalized output for continuous columns
    logits_cont = torch.matmul(dis_inputs, G_W3[:, :n_con_cols]) + G_b3[:, :n_con_cols]
    G_prob_cont = torch.sigmoid(logits_cont)

    # Concatenate categorical and continuous columns
    G_prob = torch.cat((G_prob_cat, G_prob_cont), dim=1)
    logits = torch.cat((logits_cat, logits_cont), dim=1)

    return G_prob'''

############### ORIGINAL CODE ###################################################
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

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
'''import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain (data_x, gain_parameters):
  Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
            
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)
          
  return imputed_data'''