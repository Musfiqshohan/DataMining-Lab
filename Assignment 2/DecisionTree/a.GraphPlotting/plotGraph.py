


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_graph(xaxis, y1axis, y2axis, measure):
    n_groups = len(xaxis)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, y1axis, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Decision Tree')

    rects2 = plt.bar(index + bar_width, y2axis, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Naive Bayes')

    plt.xlabel('Dataset')
    plt.ylabel(measure)
    plt.title('Evaluation measures: '+measure)
    plt.xticks(index + bar_width, xaxis, rotation=25, fontsize=8)
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.savefig(measure+".png")






decisiontree={}
filename = "decisiontreemeasures.txt"
with open(filename) as f:
    for line in f:
        attributes = line.rstrip().split(',')
        name,acc,pre,rec,f1,time=attributes
        decisiontree[name]=[float(acc)*100,float(pre)*100,float(rec)*100,float(f1)*100,float(time)]

naivebayes={}
filename = "bayesmeasures.txt"
with open(filename) as f:
    for line in f:
        attributes = line.rstrip().split(',')
        name,acc,pre,rec,f1,time=attributes
        naivebayes[name]=[float(acc),float(pre),float(rec)*100,float(f1)*100,float(time)/1000]



xaxis=decisiontree.keys()
y1axis,y2axis=[],[]

for dataset in decisiontree:
    y1axis.append( decisiontree[dataset][0])
    y2axis.append( naivebayes[dataset][0])

plot_graph(xaxis,y1axis,y2axis,"Accuracy")

#
# for id in range(len(measure)):
#     for dataset in decisiontree:
#         y1axis.append( decisiontree[dataset][id])
#         y2axis.append( naivebayes[dataset][id])
#
#         plot_graph(xaxis,y1axis,y2axis,measure[id])

