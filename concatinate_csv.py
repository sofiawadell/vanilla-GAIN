import pandas as pd
import numpy as np


data_bank = pd.read_csv('results/optimal_hyperparameters_GAIN_gain_v2_bank.csv')
data_credit = pd.read_csv('results/optimal_hyperparameters_GAIN_gain_v2_credit.csv')
data_letter = pd.read_csv('results/optimal_hyperparameters_GAIN_gain_v2_letter.csv')
data_mushroom_10 = pd.read_csv('results/optimal_hyperparameters_GAIN_gain_v2_mushroom_10.csv')
data_mushroom = pd.read_csv('results/optimal_hyperparameters_GAIN_gain_v2_mushroom.csv')
data_news = pd.read_csv('results/optimal_hyperparameters_GAIN_gain_v2_news.csv')

data_concat = pd.concat([data_mushroom_10, data_mushroom, data_letter, data_bank, data_credit, data_news], ignore_index=True)
data_concat.to_csv('results/optimal_hyperparameters_GAIN_gain_v2.csv', index=False)
print(data_concat)