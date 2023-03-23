import pandas as pd
import numpy as np
from utils import normalization, renormalization

data_name = "news"
missingness = 10

filename_train = '{}{}_{}_{}.csv'.format('preprocessed_data/one_hot_train_data_wo_target/one_hot_', data_name, 'train', missingness)
filename_test = '{}{}_{}_{}.csv'.format('preprocessed_data/one_hot_test_data_wo_target/one_hot_', data_name, 'test', missingness)

df_train = pd.read_csv(filename_train)
df_test = pd.read_csv(filename_test)

cat_cols = ['data_channel', 'weekday', 'is_weekend']
count = []

for i in range(len(cat_cols)):
    count.append(sum(df_train.columns.str.contains(cat_cols[i], regex=True)))

cat_cols = {cat_cols[i]: count[i] for i in range(len(cat_cols))}
print(cat_cols)

