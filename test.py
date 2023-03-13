import pandas as pd
from datasets import datasets

# Determine dataset, missingness and mode (test/train)
all_datasets = ["mushroom", "news", "credit", "letter", "bank"]
all_missingness = [10, 30, 50, 70]


for data_name in all_datasets:
    for missingness in all_missingness:
        target = datasets[data_name]["target"]
        filename_train = '{}{}_{}_{}.csv'.format('one_hot_train_data/one_hot_', data_name, 'train', missingness)
        filename_test = '{}{}_{}_{}.csv'.format('one_hot_test_data/one_hot_', data_name, 'test', missingness)

        df_train = pd.read_csv(filename_train)
        df_test = pd.read_csv(filename_test)

        df_train = df_train.drop(target, axis=1)
        df_test = df_test.drop(target, axis=1)

        # Save as csv
        filename_train = '{}{}_{}_{}.csv'.format('one_hot_train_data_wo_target/one_hot_', data_name, 'train', missingness)
        df_train.to_csv(filename_train, index=False)

        filename_test = '{}{}_{}_{}.csv'.format('one_hot_test_data_wo_target/one_hot_', data_name, 'test', missingness)
        df_test.to_csv(filename_test, index=False)




