import pandas as pd
import separate_segmentation
import sort_segmentation

# Reading dataset
dataset = pd.read_csv('../../Dataset/Cleaned_Dataset.csv')

# Creating a Column for Frequency
dataset['Number_of_Books'] = dataset.apply(
    lambda row: int(round(row['Book_length(mins)_overall'] / row['Book_length(mins)_avg'])), axis=1)

# Setting RFM factors
R = 'Last_Visited_mins_Purchase_date'
F = 'Number_of_Books'
M = 'Price_overall'

# Performing RFM Separate Segmentation
dataset = separate_segmentation.segmentation(dataset, [R, F, M], 3)

# Calculating size of each segment
data = dataset.groupby('Separate_Approch_Segment')['id'].nunique()

# Adding empty segments
for i in range(1, 28):
    if i not in data.index:
        data = data.append(pd.Series([0,], index=[i,]))

# Changing indices to 3 character format
new_index = []
for i in range(1, 4):
    for j in range(1, 4):
        for k in range(1, 4):
            new_index.append(str(i) + str(j) + str(k))
data = data.sort_index()
data.index = new_index

# Plotting segments using bar plot
data.plot.bar(color='red', xlabel='Segments', ylabel='Frequency', title='Separate_RFM_Segmentation')

# Performing RFM Sort Segmentation
dataset = sort_segmentation.segmentation(dataset, [R, F, M], 3)

# Calculating F Mean for each segment
data1 = dataset.groupby("Sort_Approch_Segment")[F].mean()

# Changing indices to 3 character format
data1.index = new_index

# Plotting segments F mean using bar plot
data1.plot.bar(color='green', xlabel='Segments', ylabel='Mean of F', title='Sort_RFM_Segmentation')

# Calculating M Mean for each segment
data2 = dataset.groupby("Sort_Approch_Segment")[M].mean()

# Changing indices to 3 character format
data2.index = new_index

# Plotting segments M Mean using bar plot
data2.plot.bar(color='orange', xlabel='Segments', ylabel='Mean of M', title='Sort_RFM_Segmentation')

# Output: Dataset with two segmentation columns
dataset.to_csv('RFM.csv', index=False)
