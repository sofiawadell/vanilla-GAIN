import pandas as pd
import numpy as np
from datasets import datasets

''' Description

Two datasets, one complete and one incomplete. Remove target variable and then do one-hot-encoding
where the whole row for a certain feature is replaced with NaN if the value was missing in the no one-hot-encoded
dataset. 

'''
# Determine dataset, missingness and mode (test/train)
#all_datasets = ["mushroom", "news", "credit", "letter", "bank"]
all_datasets = ["credit"]
all_missingness = [10, 30, 50, 70]

dataset = "credit"
missingness = 70

# Find categorical and target column
cat_cols = datasets[dataset]["cat_cols"]
target_col = datasets[dataset]["target"]

# Concatenate complete datasets
filename_train_complete = 'train_data/{}_train.csv'.format(dataset)
train_data_complete = pd.read_csv(filename_train_complete)
train_data_complete = train_data_complete.drop(target_col, axis=1)

filename_test_complete = 'test_data/{}_test.csv'.format(dataset)
test_data_complete = pd.read_csv(filename_test_complete)
test_data_complete = test_data_complete.drop(target_col, axis=1)

full_data_complete = pd.concat([train_data_complete, test_data_complete], axis=0)

# Concatenate datasets with missingness
filename_train_x = 'train_data/{}_train_{}.csv'.format(dataset, missingness)
train_data_x = pd.read_csv(filename_train_x)
train_data_x = train_data_x.drop(target_col, axis=1)

filename_test_x = 'test_data/{}_test_{}.csv'.format(dataset, missingness)
test_data_x = pd.read_csv(filename_test_x)
test_data_x = test_data_x.drop(target_col, axis=1)

full_data_x = pd.concat([train_data_x, test_data_x], axis=0)

# Create copy of dataframes
df_full_data_complete = full_data_complete.copy()
df_full_data_x = full_data_x.copy()

# Loop through each categorical column and apply one-hot encoding
for col in cat_cols:
    # Get unique categories across both datasets
    categories = full_data_complete[col].unique()

    # Check if each category is present in the missing dataset
    missing_categories = set(categories) - set(full_data_x[col].unique())

    # Perform one-hot encoding on the column, specifying the column order and feature name prefix
    prefix = col
    encoded_col_missing = pd.get_dummies(full_data_x[col], prefix=prefix, columns=categories)
    encoded_col_complete = pd.get_dummies(full_data_complete[col], prefix=prefix, columns=categories)

    # Add new columns to missing dataset for any missing categories
    for category in missing_categories:
        prefix = col
        new_col = pd.Series([0] * len(full_data_x))
        new_col.name = prefix + str(category)
        #full_data_x = pd.concat([full_data_x, new_col], axis=1)
        encoded_col_missing[new_col.name] = new_col

    # Replace any rows with missing values with NaN
    encoded_col_missing[full_data_x[col].isna()] = pd.NA
                    
    # Add the encoded column(s) to the new dataframe
    df_full_data_complete = pd.concat([df_full_data_complete, encoded_col_complete], axis=1)
    df_full_data_x  = pd.concat([df_full_data_x, encoded_col_missing], axis=1)

# Remove the original categorical columns from the new dataframe
df_full_data_x.drop(cat_cols, axis=1, inplace=True)
df_full_data_complete.drop(cat_cols, axis=1, inplace=True)

# Split back into training and test
train_data_complete, test_data_complete = np.vsplit(df_full_data_complete, [len(train_data_complete)])
train_data_x, test_data_x = np.vsplit(df_full_data_x, [len(train_data_x)])

# Save to CSV
filename_train_complete = 'one_hot_train_data/one_hot_{}_train.csv'.format(dataset)
train_data_complete.to_csv(filename_train_complete, index=False)
filename_test_complete = 'one_hot_test_data/one_hot_{}_test.csv'.format(dataset)
test_data_complete.to_csv(filename_test_complete, index=False)

filename_train_x = 'one_hot_train_data/one_hot_{}_train_{}.csv'.format(dataset, missingness)
train_data_x.to_csv(filename_train_x, index=False)
filename_test_x = 'one_hot_test_data/one_hot_{}_test_{}.csv'.format(dataset, missingness)
test_data_x.to_csv(filename_test_x, index=False)

print(test_data_x.shape)
print(train_data_x.shape)

print(test_data_complete.shape)
print(train_data_complete.shape)
