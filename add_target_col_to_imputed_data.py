import numpy as np
import pandas as pd
from datasets import datasets

# Add target column to imputed data
data_name = "credit"
miss_rate = 10

filename = 'preprocessed_data/one_hot_test_data/one_hot_{}_test_{}.csv'.format(data_name, miss_rate)
full_data = pd.read_csv(filename)

target_column = full_data.iloc[:, -1]

filename_imputed = 'imputed_data/{}_{}_wo_target.csv'.format(data_name, miss_rate)
imputed_data = pd.read_csv(filename_imputed)

imputed_data[datasets[data_name]["target"]] = target_column.values

filename_with_target = 'imputed_data/{}_{}.csv'.format(data_name, miss_rate)
imputed_data.to_csv(filename_with_target, index=False)