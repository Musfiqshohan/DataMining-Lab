import networkx as nx
import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import write_dot'','' graphviz_layout
import random
from sklearn import svm
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from numpy import array




import numpy as np

year = [2014, 2015, 2016, 2017, 2018, 2019]
tutorial_count = [39, 117, 111, 110, 67, 29]

plt.plot(year, tutorial_count, color="#6c3376", linewidth=3)
plt.xlabel('Year')
plt.ylabel('Number of futurestud.io Tutorials')
plt.savefig('line_plot.pdf')