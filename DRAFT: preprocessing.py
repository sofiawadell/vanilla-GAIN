import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datasets import datasets 


df = pd.read_csv('preprocessed_data/bank.csv')
unknown_count = df.isin(['unknown']).sum().sum()
total_num_missing = df.isnull().sum().sum()

print(unknown_count)
print(total_num_missing)



""" df = pd.read_csv('preprocessed_data/letter.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv('preprocessed_data/letter_train.csv', index=False)
test_data.to_csv('preprocessed_data/letter_test.csv', index=False)
 """
#cat_cols = datasets["bank"]["cat_cols"]
#num_cols = datasets["bank"]["num_cols"]
#target_col = datasets["bank"]["target"]

# Fill missing values with "unknown"
#df[cat_cols] = df[cat_cols].fillna('unknown')

# Drop target column
#df = df.drop(target_col, axis=1)

# Drop drop column
#df = df.drop(drop_cols, axis=1)
""" 
# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='if_binary', sparse=False)
cat_encoded = encoder.fit_transform(df[cat_cols])

# Concatenate the one-hot encoded categorical columns with the numerical columns
df = pd.concat([df[num_cols], pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))], axis=1)

df.to_csv('preprocessed_data/bank.csv', index=False) """