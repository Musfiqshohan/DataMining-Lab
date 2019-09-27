import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import random
from sklearn import svm
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from numpy import array


a=[1,1,2,2]
a=set(a)
print(list(a))


#
# def getData(dataset, indexList):
#     indexList=array(indexList)
#     D=dataset.iloc[indexList,:]
#     return D
#
# dataset = pandas.read_csv('Dataset/iris.data')
#
# singletup= dataset.iloc[1,:]
# dimension=len(singletup)-1
# colList=[x for x in range(dimension)]
# classlabList=[dimension]
# X = dataset.iloc[:, colList]
# y = dataset.iloc[:, classlabList]
#
#
# best_svr = SVR(kernel='rbf')
# cv = KFold(n_splits=10, random_state=42, shuffle=False)
# for train_index, test_index in cv.split(X):
#     print("Train Index: ", train_index, "\n")
#     print("Test Index: ", test_index)
#
#     X_train, X_test, y_train, y_test = getData(X,train_index), getData(X,test_index)\
#         , getData(y,train_index), getData(y,test_index)
#
#     print(X_train)
#     print(X_test)
