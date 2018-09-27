#!/usr/bin/env python
import numpy as np
import pandas as pd
import graphviz

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modules.utils import binary_target, replaceNanWithAverage

# Load datas
data = pd.read_csv('data/uci-heart-disease/processed_cleveland_data.csv', sep=',')
print 'Shape dataset: ', data.shape
print data.head()

# Process datas
x = data.values[:, 0:13]
x = replaceNanWithAverage(x)
y = data.values[:, -1]
y = np.array(list(map(binary_target, y)))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=100)

# Model C4.5
c = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100, 
                                max_depth=13, min_samples_leaf=2)

# Train model
c.fit(x_train, y_train)

# Metric - Predict
y_pred_train = c.predict(x_train)
print "acc train: ", accuracy_score(y_train, y_pred_train) * 100
y_pred_test = c.predict(x_test)
print "acc test: ", accuracy_score(y_test, y_pred_test) * 100
print y_pred_test

#Plot rules
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target_names = ['0', '1']
dot_data = tree.export_graphviz(c, out_file=None, 
                                feature_names=feature_names,  
                                class_names=target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph

print ("Done!")
