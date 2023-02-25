import pandas as pd


# Load data file into a pandas dataframe with header
header = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17']  # Replace with your own header row
df = pd.read_csv('data/letter-re.data', header=None, names=header)

# Save dataframe to CSV file with comma delimiter
df.to_csv('data/output.csv', index=False, sep=',')