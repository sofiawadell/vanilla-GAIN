import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datasets import datasets 


""" df = pd.read_csv('preprocessed_data/letter.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv('preprocessed_data/letter_train.csv', index=False)
test_data.to_csv('preprocessed_data/letter_test.csv', index=False)
 """

# news ok, inte credit, inte mushroom
dataset = 'mushroom'

cat_cols = datasets[dataset]["cat_cols"]
num_cols = datasets[dataset]["num_cols"]
target_col = datasets[dataset]["target"]

df_test = pd.read_csv('one_hot_test_data/one_hot_'+dataset+'_test_50.csv')
df_train = pd.read_csv('one_hot_train_data/one_hot_'+dataset+'_train_50.csv')

print(df_test.shape)
print(df_train.shape)

df_test_complete = pd.read_csv('one_hot_test_data/one_hot_'+dataset+'_test.csv')
df_train_complete = pd.read_csv('one_hot_train_data/one_hot_'+dataset+'_train.csv')

print(df_test_complete.shape)
print(df_train_complete.shape)



#df.to_csv('original_data_num_first/'+dataset+'.csv', index=False)

""" 
# Fill missing values with "unknown"
#df[cat_cols] = df[cat_cols].fillna('unknown')

# Drop target column
#df = df.drop(target_col, axis=1)

# Drop drop column
#df = df.drop(drop_cols, axis=1)


# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='if_binary', sparse=False)
cat_encoded = encoder.fit_transform(df[cat_cols])

# Concatenate the one-hot encoded categorical columns with the numerical columns
df = pd.concat([df[num_cols], pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))], axis=1)

df.to_csv('original_data_num_first/bank.csv', index=False) """