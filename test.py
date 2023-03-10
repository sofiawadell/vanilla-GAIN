import pandas as pd


data_name = "news"

df_train = pd.read_csv('train_data/'+data_name+'_train.csv')
df_test = pd.read_csv('test_data/'+data_name+'_test.csv')

size_train, dim_train = df_train.shape
size_test, dim_test = df_test.shape

print("Train: " + str(size_train))
print("Test: " + str(size_test))

print(dim_train)
print(dim_test)