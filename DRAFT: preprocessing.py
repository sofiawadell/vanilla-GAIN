import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from datasets import datasets 

df = pd.read_csv('original_data/bank.csv')

cat_cols = datasets["bank"]["cat_cols"]
num_cols = datasets["bank"]["num_cols"]
target_col = datasets["bank"]["target"]

# Fill missing values with "unknown"
#df[cat_cols] = df[cat_cols].fillna('unknown')

# Drop target column
df = df.drop(target_col, axis=1)

# Drop drop column
#df = df.drop(drop_cols, axis=1)

# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='if_binary', sparse=False)
cat_encoded = encoder.fit_transform(df[cat_cols])

# Concatenate the one-hot encoded categorical columns with the numerical columns
df = pd.concat([df[num_cols], pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))], axis=1)

df.to_csv('preprocessed_data/bank.csv', index=False)