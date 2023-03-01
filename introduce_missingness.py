import pandas as pd
import numpy as np
from utils import binary_sampler

# Introducing MCAR in dataset

data_name = "news"
miss_rate = 0.7
missingness = 70

# Read datasets
train_data_full = pd.read_csv('train_data/'+data_name+'_train.csv')
test_data_full = pd.read_csv('test_data/'+data_name+'_test.csv')
no_train, dim_train = train_data_full.shape
no_test, dim_test = test_data_full.shape

# Introduce missing training data
train_data_m = binary_sampler(1-miss_rate, no_train, dim_train)
train_miss_data_x = train_data_full.copy()
train_miss_data_x[train_data_m == 0] = np.nan

print(train_miss_data_x.isna().sum().sum())
print(train_miss_data_x.isna().sum().sum()/(no_train * dim_train))

# Introduce missing test data
test_data_m = binary_sampler(1-miss_rate, no_test, dim_test)
test_miss_data_x = test_data_full.copy()
test_miss_data_x[test_data_m == 0] = np.nan

print(test_miss_data_x.isna().sum().sum())
print(test_miss_data_x.isna().sum().sum()/(no_test * dim_test))

# Save as csv
filename_train = '{}{}_{}_{}.csv'.format('train_data/', data_name, 'train', missingness)
train_miss_data_x.to_csv(filename_train, index=False)

filename_test = '{}{}_{}_{}.csv'.format('test_data/', data_name, 'test', missingness)
test_miss_data_x.to_csv(filename_test, index=False)