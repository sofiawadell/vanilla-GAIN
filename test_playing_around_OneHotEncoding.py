import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datasets import datasets
from utils import binary_sampler

dataset = "bank"
df = pd.read_csv('one_hot_train_data/one_hot_'+dataset+'_train_70.csv')

df = df.to_numpy()

print(df)

#print(df.get("campaign").dtype)


'''print(df.head())

# Categorical and numerical columns
cat_cols = datasets[dataset]["cat_cols"]
num_cols = datasets[dataset]["num_cols"]
no, dim = df.shape

# Fill missing values with "unknown"
#df[cat_cols] = df[cat_cols].fillna('unknown')

# Drop target column
#df = df.drop(target_col, axis=1)

# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='error')
cat_encoded = encoder.fit_transform(df[cat_cols])

# Concatenate the one-hot encoded categorical columns with the numerical columns
df_encoded = pd.concat([df[num_cols], pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))], axis=1)

print(df_encoded.head())

#################################################

#Introduce missingness
miss_rate = 0.9
data_m = binary_sampler(1-miss_rate, no, dim)
miss_data_x = df.copy()
miss_data_x[data_m == 0] = np.nan

# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='if_binary', sparse=False)
cat_encoded = encoder.fit_transform(miss_data_x[cat_cols])

# Concatenate the one-hot encoded categorical columns with the numerical columns
df_miss_encoded = pd.concat([miss_data_x[num_cols], pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))], axis=1)

print(df_miss_encoded.head())'''

