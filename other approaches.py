import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
import tracemalloc
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


encode = LabelEncoder()

data = pd.read_excel("Final_data.xlsx")
#data = data.sample(frac=1)

x = data['Data']
y = data['Significance']

encode = LabelEncoder()
y = encode.fit_transform(y)

all_data = np.array(x)

#binary encoded (one hot encoder)
Onehot_encoder = OneHotEncoder(sparse=False)
all_data = all_data.reshape(len(all_data), 1)
encoded_x = Onehot_encoder.fit_transform(all_data)

x_train, x_test, y_train, y_test = train_test_split(encoded_x, y, test_size= 0.15, random_state=21)

# Support vector machine
from sklearn import svm

svm = svm.SVC(gamma='scale', decision_function_shape='ovo')
svm.fit(x_train, y_train)
prediction5 = svm.predict(x_test)
print("Accuracy of Support vector machine: ",svm.score(x_test, y_test))


#naive Bayes
mnB = MultinomialNB()
mnB.fit(x_train, y_train)
prediction = mnB.predict(x_test)
print("Accuracy of Naive Bayes: ",mnB.score(x_test, y_test))


#Nearest Cendroid
from sklearn.neighbors.nearest_centroid import NearestCentroid
Nearest = NearestCentroid()
Nearest.fit(x_train, y_train)
predictions3 = Nearest.predict(x_test)
print("Accuracy of K nearest Cendroid: ",Nearest.score(x_test, y_test))


#Logistic Regression
tracemalloc.start()
from sklearn.linear_model import LogisticRegression
Logistic_regression = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, y_train)
prediction2 = Logistic_regression.predict(x_test)
print("Accuracy using Logistic Regression: ",Logistic_regression.score(x_test, y_test))


#Logistic RegressionCV
from sklearn.linear_model import LogisticRegressionCV
Logistic_regression1 = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(x_train, y_train)
prediction4 = Logistic_regression1.predict(x_test)
print("Accuracy using Logistic RegressionCV: ",Logistic_regression1.score(x_test, y_test))

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train, y_train)
print("Accuracy of K nearest Neighbor: ",neigh.score(x_test, y_test))

import graphviz
from sklearn import tree



#Decision Tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier(random_state=0)
Tree.fit(x_train, y_train)
dot_data = tree.export_graphviz(Tree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("data")
#dot_data = tree.export_graphviz(Tree, out_file=None, feature_names=data.feature_names,  class_names=data.class_names, filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.view()
print("Accuracy of Decision tree: ",Tree.score(x_test,y_test))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=43)
forest.fit(x_train, y_train)
prediction7 = forest.predict(x_test)
print("Accuracy of Random Forest: ",forest.score(x_test, y_test))