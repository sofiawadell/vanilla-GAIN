import pandas as pd
import numpy as np
from utils import normalization, renormalization

data_name = "letter"
missingness = 10

filename_train = '{}{}_{}_{}.csv'.format('one_hot_train_data_wo_target/one_hot_', data_name, 'train', missingness)
filename_test = '{}{}_{}_{}.csv'.format('one_hot_test_data_wo_target/one_hot_', data_name, 'test', missingness)

df_train = pd.read_csv(filename_train)
df_test = pd.read_csv(filename_test)

np_array = df_train.values

norm_data, norm_params = normalization(np_array)
max_values = np.nanmax(norm_data, axis=0)
min_values = np.nanmin(norm_data, axis=0)

print(max_values)
print(min_values)

re_norm_data = renormalization(norm_data, norm_params)

if np.array_equal(re_norm_data, np_array):
    print("Correct")

