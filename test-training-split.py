import pandas as pd
from sklearn.model_selection import train_test_split



data_name = "letter"
df = pd.read_csv('original_data_num_first/'+data_name+'.csv')

#df = df.drop(columns=['ID'])

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv('train_data/'+data_name+'_train.csv', index=False)
test_data.to_csv('test_data/'+data_name+'_test.csv', index=False)

print(df.size)
print(train_data.size)
print(test_data.size)