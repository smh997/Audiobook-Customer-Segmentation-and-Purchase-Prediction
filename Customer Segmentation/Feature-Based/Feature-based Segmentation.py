import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# Reading dataset
dataset = pd.read_csv('../../Dataset/Cleaned_Dataset.csv', index_col=0)

# Checking lost values
print(dataset.isnull().sum())

# Filling lost values
dataset['Review10/10'].fillna(dataset['Review10/10'].mean(), inplace=True)
print(dataset['Review10/10'])

# Checking lost values again
print(dataset.isnull().sum())

# Dropping unnecessary data columns
dataset = dataset.drop(["id", "Target"], axis=1)

# Adding Number_of_Books column
dataset['Number_of_Books'] = dataset.apply(
    lambda row: int(round(row['Book_length(mins)_overall'] / row['Book_length(mins)_avg'])), axis=1)

# Removing Outliers

# from scipy import stats
# z_scores = stats.zscore(dataset)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# dataset = dataset[filtered_entries]


# PlottingCorrelation Matrix
correlation = dataset.corr()
plt.figure(figsize=(11, 11))
sb.heatmap(data=correlation, square=True, annot=True, vmin=-1, vmax=1, center=0,
           cmap='coolwarm', annot_kws={"size": 10}, linewidths=2, linecolor='black', )
plt.show()

# Dropping some additional columns and redraw matrix
dataset.drop(['Book_length(mins)_overall', 'Book_length(mins)_avg', 'Price_overall', 'Review', 'Minutes_listened'],
             axis=1, inplace=True)
print(dataset.columns)

# PlottingCorrelation Matrix again
correlation = dataset.corr()
plt.figure(figsize=(6, 6))
sb.heatmap(data=correlation, square=True, annot=True, vmin=-1, vmax=1, center=0,
           cmap='coolwarm', annot_kws={"size": 10}, linewidths=2, linecolor='black',)
plt.show()

# Normalization
x = dataset.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
dataset = pd.DataFrame(x_scaled)
print('Normalized:')
print(dataset)

# Plotting Elbow Method
distortions = []
inertias = []
distortion_mapping = {}
inertia_mapping = {}
K = range(1, 11)

for k in K:
    # Building and fitting the model for each k
    X = dataset.values
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)

    # Calculating distortion and inertia
    distortion = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
    distortions.append(distortion)
    inertias.append(kmeanModel.inertia_)
    distortion_mapping[k] = distortion
    inertia_mapping[k] = kmeanModel.inertia_

# Distortion of each k
print('Distortion of each k:')
for k, d in distortion_mapping.items():
    print(f'{k} : {d}')

# Distortion Elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

# Inertia of each k
print('Inertia of each k:')
for k, i in inertia_mapping.items():
    print(f'{k} : {i}')

# Inertia Elbow
plt.plot(K, inertias, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()


# Clustering

# Initializing and using PCA
pca = PCA(2)
df = pca.fit_transform(dataset)

# Perform K-Means with suitable k = 3
kmeans = KMeans(n_clusters=3)
kmeans_model = kmeans.fit(df)
label = kmeans_model.labels_

# Calculating Silhouette Score and Inertia for analyzing our KMeans performance
print('Silhouette Score:', metrics.silhouette_score(df, label, metric='euclidean'))
print('Inertia:', kmeans.inertia_)

# Getting unique labels for PCA
u_labels = np.unique(label)

# Plotting the results (PCA):
for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
plt.legend()
plt.show()

# Getting the Centroids for PCA
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

# plotting the Centroids (PCA):
for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='black')
plt.legend()
plt.show()
