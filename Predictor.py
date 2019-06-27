import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

my_csv = pd.read_csv(
    'G:/RedBufferExercise/champs-scalar-coupling/train.csv', nrows=230)
structure = pd.read_csv(
    'G:/RedBufferExercise/champs-scalar-coupling/structures.csv', nrows=123)

feature_vector = []
column1 = my_csv['atom_index_0']
column2 = my_csv['atom_index_1']

m_names = my_csv['molecule_name']
m_names2 = structure['molecule_name']

names = np.array(list(sorted(set(list(m_names)))))
names2 = np.array(list(sorted(set(list(m_names2)))))

# print (column1,column2)
# print(names,names2)

list_names = []

for i in range(len(names2)):
    list_n = my_csv.loc[my_csv['molecule_name']
                        == names2[i], 'molecule_name'].values
    list_names.append(len(list_n))

number = list_names
list_names = np.cumsum(list_names)

list_names = np.roll(list_names, 1)
for i in range(len(number)):
    if number[i] == 0:
        list_names[i] = 0

# number = np.roll(number,1)
list_names[0] = 0
print(number, list_names)

for j in range(len(list_names)):
    for i in range(0, number[j]):

        skiprows = [k for k in range(1, list_names[j]+1)]
        structure = pd.read_csv(
            'G:/RedBufferExercise/champs-scalar-coupling/structures.csv', nrows=300, skiprows=skiprows)
        inc = list_names[j]
        if (j == 0):

            x1 = structure.loc[structure['atom_index'] ==
                               column1[i], 'x'].values[0]
            x2 = structure.loc[structure['atom_index'] ==
                               column2[i], 'x'].values[0]

            y1 = structure.loc[structure['atom_index'] ==
                               column1[i], 'y'].values[0]
            y2 = structure.loc[structure['atom_index'] ==
                               column2[i], 'y'].values[0]

            z1 = structure.loc[structure['atom_index']
                               == column1[i], 'z'].values[0]
            z2 = structure.loc[structure['atom_index']
                               == column2[i], 'z'].values[0]
            feature_vector.append(euclidean_distances(
                np.array([x1, y1, z1]).reshape(1, -1), np.array([x2, y2, z2]).reshape(1, -1)))
        if (inc != 0 and j != 0):
            x1 = structure.loc[structure['atom_index']
                               == column1[i+inc], 'x'].values[0]
            x2 = structure.loc[structure['atom_index']
                               == column2[i+inc], 'x'].values[0]

            y1 = structure.loc[structure['atom_index']
                               == column1[i+inc], 'y'].values[0]
            y2 = structure.loc[structure['atom_index']
                               == column2[i+inc], 'y'].values[0]

            z1 = structure.loc[structure['atom_index']
                               == column1[i+inc], 'z'].values[0]
            z2 = structure.loc[structure['atom_index']
                               == column2[i+inc], 'z'].values[0]

            feature_vector.append(euclidean_distances(
                np.array([x1, y1, z1]).reshape(1, -1), np.array([x2, y2, z2]).reshape(1, -1)))

n = 5

# Reshaping the feature vectors

FV = np.array(feature_vector).reshape(-1, 1)
labels = np.array(my_csv['scalar_coupling_constant']).reshape(-1, 1)

FV = FV[1:n]
labels = labels[1:n]

# Preprocessing i.e. Scaling the data

labels = preprocessing.scale(labels)
FV = preprocessing.scale(FV)

# Train- Test splitting the data

X_train, X_test, y_train, y_test = train_test_split(
    FV, labels, test_size=0.30, random_state=42)

# Training Support Vector Regression model


svr = SVR(kernel='linear', gamma=2, C=2 ^ 7)
svr.fit(X_train, y_train.reshape(-1))

# Predicting on test

p_labels = svr.predict(X_test)


print(svr.score(y_test.reshape(-1, 1), p_labels.reshape(-1, 1)))

reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train, y_train))
print(FV, labels)
