import pandas as pd
from datasets import datasets

''' Description

Two datasets, one complete and one incomplete. Remove target variable and then do one-hot-encoding
where the whole row for a certain feature is replaced with NaN if the value was missing in the no one-hot-encoded
dataset. 

'''
# Determine dataset, missingness and mode (test/train)
all_datasets = ["mushroom", "news", "credit", "letter", "bank"]
all_missingness = [10, 30, 50, 70]
modes = ["train", "test"]

for missingness in all_missingness:
        for dataset in all_datasets:
            for mode in modes:
                cat_cols = datasets[dataset]["cat_cols"]
                target_col = datasets[dataset]["target"]

                filename_incomplete = '{}_{}{}_{}_{}.csv'.format(mode, 'data/', dataset, mode, missingness)
                missing_data = pd.read_csv(filename_incomplete)

                filename_complete = '{}_{}{}_{}.csv'.format(mode, 'data/', dataset, mode)
                complete_data = pd.read_csv(filename_complete)

                # Drop target column
                complete_data = complete_data.drop(target_col, axis=1)
                missing_data = missing_data.drop(target_col, axis=1)

                # Create copy of dataframes
                df_missing_encoded = missing_data.copy()
                df_complete_encoded = complete_data.copy()
                    
                # Loop through each categorical column and apply one-hot encoding
                for col in cat_cols:
                    # Get unique categories across both datasets
                    filename_complete = 'original_data_num_first/'+dataset+'.csv'
                    complete_data_full = pd.read_csv(filename_complete)
                    categories = pd.concat([complete_data[col], complete_data_full[col]]).unique()
                    categories.sort()

                    # Perform one-hot encoding on the column, specifying the column order and feature name prefix
                    prefix = col + '_'
                    encoded_col_missing = pd.get_dummies(missing_data[col], prefix=prefix, columns=categories)
                    encoded_col_missing = encoded_col_missing.reindex(columns=[prefix+str(c) for c in categories], fill_value=0)
                    encoded_col_complete = pd.get_dummies(complete_data[col], prefix=prefix, columns=categories)
                    encoded_col_complete = encoded_col_complete.reindex(columns=[prefix+str(c) for c in categories], fill_value=0)

                    # Replace any rows with missing values with NaN
                    encoded_col_missing[missing_data[col].isna()] = pd.NA
                    
                    # Add the encoded column(s) to the new dataframe
                    df_missing_encoded = pd.concat([df_missing_encoded, encoded_col_missing], axis=1)
                    df_complete_encoded = pd.concat([df_complete_encoded, encoded_col_complete], axis=1)

                # Remove the original categorical columns from the new dataframe
                df_missing_encoded.drop(cat_cols, axis=1, inplace=True)
                df_complete_encoded.drop(cat_cols, axis=1, inplace=True)

                # Save to CSV
                save_filename_missing = '{}{}_{}{}_{}_{}.csv'.format('one_hot_', mode, 'data/one_hot_', dataset, mode, missingness)
                df_missing_encoded.to_csv(save_filename_missing, index=False)
                save_filename_complete = '{}{}_{}{}_{}.csv'.format('one_hot_', mode, 'data/one_hot_', dataset, mode)
                df_complete_encoded.to_csv(save_filename_complete, index=False)



'''# One-hot encode the categorical columns in the incomplete dataset
encoded_dataset = pd.get_dummies(incomplete_dataset, columns=cat_cols)

# Reindex the encoded dataset with all categories from the complete dataset
encoded_dataset = encoded_dataset.reindex(columns=complete_dataset.columns, fill_value=0)

# Replace rows with missing values with NaN
encoded_dataset[incomplete_dataset.isnull().any(axis=1)] = pd.NA

print(encoded_dataset.head())


###############################

# Create copy of dataframe  
df_encoded = df.copy()

# Loop through each categorical column and apply one-hot encoding
for col in cat_cols:
    # Perform one-hot encoding on the column
    encoded_col = pd.get_dummies(df[col], prefix=col)
    
    # Replace any rows with missing values with NaN
    encoded_col[df[col].isna()] = pd.NA
    
    # Add the encoded column(s) to the new dataframe
    df_encoded = pd.concat([df_encoded, encoded_col], axis=1)
    
# Remove the original categorical columns from the new dataframe
df_encoded.drop(cat_cols, axis=1, inplace=True)

print(df_encoded.head())

# Make sure still same missingness
no_missing_inital = df.isnull().sum().sum()
no, dim = df.shape
total_size = no * dim

print(no_missing_inital)
print(total_size)
print (no_missing_inital/total_size)

# df_encoded.to_csv('OLD_preprocessed_data/'+dataset+'.csv', index=False)


#['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                     #'day_of_week', 'poutcome'], 

#alternatives = df['month'].unique()'''
