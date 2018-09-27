#!/usr/bin/env python
import numpy as np
import pandas as pd
import graphviz

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modules.utils import binary_target, replaceNanWithAverage

data = pd.read_csv('data/uci-heart-disease/processed_cleveland_data.csv', sep=',')
print 'Length dataset: ', len(data)
print 'Shape dataset: ', data.shape
print data.head()

x = data.values[:, 0:13]
x = replaceNanWithAverage(x)
y = data.values[:, -1]
y = np.array(list(map(binary_target, y)))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=100)

# Train model
c = tree.DecisionTreeClassifier()
c.fit(x_train, y_train)

acc_train = np.sum(c.predict(x_train) == y_train) / float(y_train.size)
acc_test = np.sum(c.predict(x_test) == y_test) / float(y_test.size)
print("acc train:", acc_train)
print("acc test:", acc_test)

#Plot rules
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target_names = ['no presence', 'presence']
dot_data = tree.export_graphviz(c, out_file=None, 
                                feature_names=feature_names,  
                                class_names=target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph

print ("Done!")
