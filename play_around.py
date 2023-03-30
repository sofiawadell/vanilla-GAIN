import pandas as pd
import numpy as np
from utils import normalization, renormalization

data_name = "letter"
missingness = 10

filename_train = '{}{}_{}_{}.csv'.format('preprocessed_data/one_hot_train_data_wo_target/one_hot_', data_name, 'train', missingness)
filename_test = '{}{}_{}_{}.csv'.format('preprocessed_data/one_hot_test_data_wo_target/one_hot_', data_name, 'test', missingness)

df_train = pd.read_csv(filename_train)
df_test = pd.read_csv(filename_test)

no, dim = df_train.shape

print(dim)

