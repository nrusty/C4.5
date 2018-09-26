#!/usr/bin/env python
import numpy as np
import pandas as pd
import graphviz

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load model
data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, 
                                                    test_size=.20, random_state=42)
#print (data.data, data.target)

# Train model
c = tree.DecisionTreeClassifier()
c.fit(x_train, y_train)

acc_train = np.sum(c.predict(x_train) == y_train) / float(y_train.size)
acc_test = np.sum(c.predict(x_test) == y_test) / float(y_test.size)
print("acc train:", acc_train)
print("acc test:", acc_test)

# Plot rules
dot_data = tree.export_graphviz(c, out_file=None, 
                                feature_names=data.feature_names,  
                                class_names=data.target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph

print ("Done!")
