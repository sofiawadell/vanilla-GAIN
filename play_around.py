import pandas as pd
import numpy as np
from utils import normalization, renormalization

data_name = "letter"
miss_rate = 10
extra_amount = 50

filename_miss = 'preprocessed_data/one_hot_train_data_wo_target_extra_{}/one_hot_{}_train_{}_extra_{}.csv'.format(extra_amount, data_name, miss_rate, extra_amount)
filename_full = 'preprocessed_data/one_hot_train_data_wo_target_extra_{}/one_hot_{}_train_full{}_extra_{}.csv'.format(extra_amount, data_name, miss_rate, extra_amount)

df_miss = pd.read_csv(filename_miss)
df_full = pd.read_csv(filename_full)

no, dim = df_miss.shape
no_full, dim_full = df_full.shape

print(no)
print(no_full)

###########
filename_miss = 'train_test_split_data/train_data_wo_target_extra_50/letter_train_10_extra_50.csv'
filename_full = 'train_test_split_data/train_data_wo_target_extra_50/letter_train_full10_extra_50.csv'

df_miss = pd.read_csv(filename_miss)
df_full = pd.read_csv(filename_full)

no, dim = df_miss.shape
no_full, dim_full = df_full.shape

print(no)
print(no_full)

