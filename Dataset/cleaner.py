import pandas as pd

# Reading dataset
dataset = pd.read_csv('Dataset.csv')

# Modifying columns
dataset = dataset.rename(columns={"Book_length(mins)_overall": "Book_length(mins)_avg",
                                  'Book_length(mins)_avg': 'Book_length(mins)_overall',
                                  'Price_overall': 'Price_avg', 'Price_avg': 'Price_overall'})

# Correcting Review Column
dataset.loc[dataset['Review10/10'].isnull() == False, 'Review'] = 1

# Output: Cleaned Dataset
dataset.to_csv('Cleaned_Dataset.csv', index=False)
