import random
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Reading dataset
dataset = pd.read_csv('../../Dataset/Cleaned_Dataset.csv', index_col=0)

# Filling lost values
dataset['Review10/10'].fillna(dataset['Review10/10'].mean(), inplace=True)

# Saving a copy of dataset
unpreprocessed_dataset = dataset.copy()

# Dropping unnecessary data columns (Feature Selection)
dataset = dataset.drop(["id", "Target", 'Book_length(mins)_overall', 'Book_length(mins)_avg',
                        'Price_overall', 'Review', 'Minutes_listened'], axis=1)

# Adding Number_of_Books column
dataset['Number_of_Books'] = dataset.apply(
    lambda row: int(round(row['Book_length(mins)_overall'] / row['Book_length(mins)_avg'])), axis=1)

# Normalization
x = dataset.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
dataset = pd.DataFrame(x_scaled)

# Balancing

np_dataset = unpreprocessed_dataset.values
unpreprocessed_data = np_dataset[:, 1:-1]
targets = np_dataset[:, -1]
size = targets.shape[0]
one_cnt = int(np.sum(targets))
zero_cnt = size - one_cnt
zero_indices = []
remove_indices = []

# Removing additional zero labeled data in random
for i in range(size):
    if targets[i] == 0:
        zero_indices.append(i)
for i in range((zero_cnt - one_cnt)):
    r_index = random.randint(0, zero_cnt-i-1)
    remove_indices.append(zero_indices[r_index])
    zero_indices.pop(r_index)
unpreprocessed_data = np.delete(unpreprocessed_data, remove_indices, axis=0)
preprocessed_data = np.delete(dataset.values, remove_indices, axis=0)
targets = np.delete(targets, remove_indices, axis=0)

# unpreprocessed_data = np.delete(unpreprocessed_data, zero_indices[zero_cnt-(zero_cnt - one_cnt):], axis=0)
# preprocessed_data = np.delete(dataset.values, zero_indices[zero_cnt-(zero_cnt - one_cnt):], axis=0)
# targets = np.delete(targets, zero_indices[zero_cnt-(zero_cnt - one_cnt):], axis=0)
# unpreprocessed_data = np.delete(unpreprocessed_data, zero_indices[:(zero_cnt - one_cnt)], axis=0)
# preprocessed_data = np.delete(dataset.values, zero_indices[:(zero_cnt - one_cnt)], axis=0)
# targets = np.delete(targets, zero_indices[:(zero_cnt - one_cnt)], axis=0)


# Shuffling
shuffled_indices = np.arange(preprocessed_data.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_data = preprocessed_data[shuffled_indices]
shuffled_targets = targets[shuffled_indices]


# Classification

df = pd.DataFrame(shuffled_data)
target_df = pd.DataFrame(shuffled_targets)
X_train, X_test, y_train, y_test = train_test_split(df, target_df, test_size=0.3, random_state=0)


# Random Forest Classifier
rf_clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
# 10-Fold Cross validation
print('Random Forest Cross Validation:', np.mean(cross_val_score(rf_clf, X_train, y_train, cv=10)))
rf_clf.fit(X_train, y_train)
predict = rf_clf.predict(X_test)

# Extreme Gradient Boost Classifier
xgb_clf = XGBClassifier(learning_rate=0.5, n_estimators=150, base_score=0.3)
# 10-Fold Cross validation
print('XGB Cross Validation:', np.mean(cross_val_score(xgb_clf, X_train, y_train, cv=10)))
xgb_clf.fit(X_train, y_train)
predict2 = xgb_clf.predict(X_test)

# Support Vector Machine Classifier
svm_clf = SVC()
# 10-Fold Cross validation
print('SVM Cross Validation:', np.mean(cross_val_score(svm_clf, X_train, y_train, cv=10)))
svm_clf.fit(X_train, y_train)
predict3 = svm_clf.predict(X_test)


def classification_data_report(name, y_test_internal, pred_internal):
    """
    A function for reporting some analytical data about classification's performance
    :param name: classification model name
    :param y_test_internal: labels of test data
    :param pred_internal: predicted labels for test data
    :return: None
    """
    print('*** {} Classification Report ***'.format(name), end='\n\n')
    print('Confusion Matrix:\n', confusion_matrix(y_test_internal, pred_internal), end='\n\n')
    print('Classification Report:\n', classification_report(y_test_internal, pred_internal), end='\n\n')
    print('Accuracy:\n', accuracy_score(y_test_internal, pred_internal), end='\n\n')


# Accuracy, F-measure and other data about classification
classification_data_report("Random Forest", y_test, predict)
classification_data_report("XGB", y_test, predict2)
classification_data_report("SVM", y_test, predict3)


# Confusion Matrix

print('Random Forest Confusion Matrix:')
plot_confusion_matrix(rf_clf, X_test, y_test, cmap="GnBu_r")
plt.show()
print()

print('XGB Confusion Matrix:')
plot_confusion_matrix(xgb_clf, X_test, y_test, cmap="GnBu_r")
plt.show()
print()

print('SVM Confusion Matrix:')
plot_confusion_matrix(svm_clf, X_test, y_test, cmap="GnBu_r")
plt.show()
print()