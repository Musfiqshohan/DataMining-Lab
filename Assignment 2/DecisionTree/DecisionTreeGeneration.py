import copy
import math
from numpy import array
import numpy as np
import itertools
from DataInput import input_data
# from DecisionTree import DecisionTreeNode
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import networkx.generators.small as gs
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np

# from DatasetLoad import load_iris_data
from DatasetLoad import load_iris_data, load_wine_data, load_breastcancer_data, load_textbook_data
from DecisionTree import DecisionTreeNode
from graphDrawing import draw_graph


# edgeLable={}
AttributeType={}
AttributeValue={}
edgeType={}
edgeLabel={}
def isSameClass(D):

    classLabels=[]
    for datatuple in D:
        classLabels.append(datatuple[1])

    classLabels=set(classLabels)
    if len(classLabels)==1:
        ret=classLabels.pop()
        return ret

    return None


def getDataColumn(attrx,Dx):

    attr=copy.deepcopy(attrx)
    D=copy.deepcopy(Dx)

    # print(attr)
    datacolumn=[]
    for tuple in D:
        # print("tuples",tuple[0], tuple[1] )
        # print("saving",tuple[0][attr],tuple[1])
        datacolumn.append((tuple[0][attr],tuple[1]))

    # print(datacolumn)
    return datacolumn

#attrInfo is a map :  22: yes
def getInfoGain(mid, attrInfox):

    attrInfo=copy.deepcopy(attrInfox)

    # print(attrInfo)
    d1len,d2len=0,0
    totLen=len(attrInfo)
    classLabel=set( [x[1] for x in attrInfo ])


    # print(classLabel)
    D1,D2={},{}
    for lab in classLabel:
        D1[lab]=0
        D2[lab]=0

    for attr in attrInfo:
        # print(attr[0],attr[1])
        if attr[0]<=mid:
            d1len+=1
            D1[attr[1]]+=1
        else:
            d2len+=1
            D2[attr[1]]+=1

    infoval1=0.0
    infoval2=0.0

    for label in classLabel:
        # print("For",label)
        if D1[label]==0:
            infoval1+=0
        else:
            x= (-D1[label]* np.log2(D1[label]/d1len))/d1len
            # print("x",x)
            infoval1+=x

        if D2[label] == 0:
            infoval2+= 0
        else:
            y= (-D2[label]* np.log2(D2[label]/d2len))/d2len
            # print("y",y)
            infoval2+=y

    # print("info1",infoval1)
    # print("info2",infoval2)
    infoval1=(d1len*infoval1)/totLen
    infoval2=(d2len*infoval2)/totLen

    return infoval1+infoval2


def getEntropy(attrInfox):

    # print(attrInfo)
    attrInfo= copy.deepcopy(attrInfox)
    attrlist=[x[0] for x in attrInfo]
    attrlist.sort()
    # print(attrlist)

    minVal=100000000000
    minMid=-1
    for id in range(len(attrlist)-1):

        mid= (attrlist[id]+attrlist[id+1])/2
        # # print(mid)
        # ### *****************need change here
        # if mid!=0.8:
        #     continue
        ret=getInfoGain(mid,attrInfo)
        # print(mid, ret)
        if ret<minVal:
            minMid=mid
            minVal=ret

    # print(minMid,"->",minVal,)
    return minMid,minVal


def getCataEntropy(attrInfox):
    attrInfo = copy.deepcopy(attrInfox)
    attrlist = set([x[0] for x in attrInfo])

    totalInfo=0
    infoLen=len(attrInfo)
    for attval in attrlist:
        print(attval)
        attvalnum=0
        d={}
        for tup in attrInfo:
            if tup[0]== attval:
                attvalnum+=1
                if tup[1] not in d:
                    d[tup[1]]=0
                d[tup[1]]+=1

        indivInfo=0
        for val in d:
            indivInfo+=  -(d[val]/attvalnum)*np.log2(d[val]/attvalnum)

        indivInfo= (attvalnum/infoLen)*indivInfo
        totalInfo+=indivInfo


    return totalInfo




def attribute_selection_method(D,attribute_list):

    # print(attribute_list)
    # attribute name, information gain, [split point]
    selected_attr=[None,100000000000,100000000000]
    for attr in attribute_list:

        # print(attr, AttributeType[attr])
        if AttributeType[attr]=="Categorical":
            gain = getCataEntropy(getDataColumn(attr, D))
            if gain < selected_attr[1]:
                selected_attr = [attr, gain, 100000000000]  #here storing information gain

        else:
            split,gain=getEntropy(getDataColumn(attr,D))
            if gain < selected_attr[1]:
                selected_attr = [attr, gain, split]

        print("attr:",attr,"gain", gain)

    return selected_attr

# retCount=0
def getMajorityVoting(Dx):

    classLabel={}
    for tuple in Dx:
        if tuple[1] not in classLabel:
            classLabel[tuple[1]]=0
        classLabel[tuple[1]]+=1

    mx=0
    mxclass=None
    for label in classLabel:
        if classLabel[label]>mx:
            mx=classLabel[label]
            mxclass=label

    # global retCount
    # retCount+=1
    return mxclass






def getPartitionsForContinuous(D, splitting_attribute,split_point):
    DatabaseList=[]
    D1 = []
    D2 = []

    for dtuple in D:

        datatuple = copy.deepcopy(dtuple)

        if datatuple[0][splitting_attribute] <= split_point:
            datatuple[0].pop(splitting_attribute, None)
            D1.append(datatuple)

        else:
            datatuple[0].pop(splitting_attribute, None)
            D2.append(datatuple)

    DatabaseList.append(D1)
    DatabaseList.append(D2)
    return DatabaseList,None

def getPartitionsForCategorical(D, splitting_attribute):
    DatabaseList = []

    # print("Before")
    # for dat in D:
    #     print(dat)

    databasedict={}
    for dtuple in D:
        datatuple = copy.deepcopy(dtuple)
        if dtuple[0][splitting_attribute] not in databasedict:
            databasedict[dtuple[0][splitting_attribute]]=[]

        datatuple[0].pop(splitting_attribute,None)
        databasedict[dtuple[0][splitting_attribute]].append(datatuple)


    # print("After printing databaes")

    for database in databasedict:
        DatabaseList.append(list(databasedict[database]))

    split_att_values=list(databasedict.keys())
    return DatabaseList,split_att_values



def Generate_decision_tree(Dx,attribute_listx):



    issameclass=isSameClass(Dx)

    # print(Dx)
    # print(attribute_listx)
    if issameclass!=None:
        obj= DecisionTreeNode(issameclass)
        # print("base case id",obj.id, obj.label)
        return obj

    if len(attribute_listx)==0:
        # global retCount
        # retCount += 1
        ret=getMajorityVoting(Dx)
        obj=DecisionTreeNode(ret)
        # print("base case id", obj.id, obj.label)
        return obj




    D= copy.deepcopy(Dx)


    attribute_list=copy.deepcopy(attribute_listx)

    splitting_attribute,infoGain, split_point = attribute_selection_method(D, attribute_list)
    # print(splitting_attribute, split_point)
    node=DecisionTreeNode(splitting_attribute)
    node.splitpoint=split_point



    attribute_list.remove(splitting_attribute)

    if AttributeType[splitting_attribute]=="Categorical":
        DatabaseList,split_att_values=getPartitionsForCategorical(D,splitting_attribute)
    else:
        DatabaseList,split_att_values=getPartitionsForContinuous(D,splitting_attribute,split_point)


    idx=0
    for partition in DatabaseList:

        if len(partition) == 0:
            childNode = getMajorityVoting(partition)
        else:
            childNode = Generate_decision_tree(partition, attribute_list)

        if AttributeType[splitting_attribute]=="Categorical":
            edgeLabel[(node,childNode)] = split_att_values[idx]
            idx += 1

        node.children.append(childNode)


    # edgeType[(node,childNode)]="less"
    # if len(D2) == 0:
    #     childNode = getMajorityVoting(D)
    # else:
    #     childNode = Generate_decision_tree(D2, attribute_list)
    # # edgeLable[splitting_attribute, childNode] = "val>%.2f" % split_point
    # node.children.append(childNode)
    # edgeType[(node,childNode)]="greater"


    return node



def runDFS(root):
    G = nx.DiGraph()

    finaledgeLabel={}
    dfs(root,G,finaledgeLabel)
    draw_graph(G,finaledgeLabel)

def dfs(node,G,finaledgeLabel):
    for child in node.children:
        u=node.label+str(node.id)
        v=child.label+str(child.id)
        G.add_edge(u,v)
        if (node,child) not in edgeLabel:
            finaledgeLabel[(u,v)]=-1
        else:
            finaledgeLabel[(u,v)]=edgeLabel[(node,child)]

        print(node.label,"->",child.label)
        dfs(child,G,finaledgeLabel)

#
# def performCrossValidation(D):
#     random.shuffle(D)
#
#



def testInDecisionTree(node,data):

    if len(node.children)==0:
        if node.label==data[1]:
            return 1
        else:
            # print("base",node.label,data[1])
            # print("Unmatched",data)
            return 0


    ret=0

    if AttributeType[node.label]=="Categorical":
        for child in node.children:
            if edgeLabel[(node,child)]==data[0][node.label]:
                ret=testInDecisionTree(child,data)
    else:
        if data[0][node.label]<= node.splitpoint:
            ret=testInDecisionTree(node.children[0],data)
        else:
            ret=testInDecisionTree(node.children[1],data)

    return ret


def transformDataSet(X_train,y_train):


    res=[]
    for tup in y_train:
        res+=tup
    y_train=res


    Dataset = []
    for tupid in range(len(X_train)):

        data = {}
        for attrId in range(len(X_train[tupid])):
            data[attribute_list[attrId]] = X_train[tupid][attrId]

        # print(data)

        Dataset.append((data, y_train[tupid]))

    return Dataset


def getData(dataset, indexList):
    indexList=array(indexList)
    D=dataset.iloc[indexList,:]
    return D



def getCategoricalAttributeValue(attribute_list,X):

    for id in range(len(attribute_list)):

        d=X.iloc[:,id]
        d= list(set(d.values.tolist()))


        AttributeValue[attribute_list[id]]=d




def runCrossValidation(attribute_list,X,y):

    covered, corrected = 0, 0
    best_svr = SVR(kernel='rbf')
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = getData(X, train_index), getData(X, test_index) \
            , getData(y, train_index), getData(y, test_index)

        trainData = transformDataSet(X_train.values.tolist(), y_train.values.tolist())
        testData = transformDataSet(X_test.values.tolist(), y_test.values.tolist())
        root = Generate_decision_tree(trainData, attribute_list)
        ret = 0
        for tup in testData:
            ret += testInDecisionTree(root, tup)
        print("res", ret, len(testData))
        covered += len(testData)
        corrected += ret
    print("Total", corrected, covered, "acuracy %.3f" % (corrected / covered * 100))



def runOnFullDataset(attribute_list,X,y):
    covered, corrected = 0, 0
    best_svr = SVR(kernel='rbf')
    cv = KFold(n_splits=2, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = getData(X, train_index), getData(X, test_index) \
            , getData(y, train_index), getData(y, test_index)

        trainData = transformDataSet(X_train.values.tolist(), y_train.values.tolist())
        testData = transformDataSet(X_test.values.tolist(), y_test.values.tolist())

        root = Generate_decision_tree(trainData+testData, attribute_list)


    runDFS(root)

if __name__ == '__main__':



    # D = input_data()
    # draw_graph(G,edgeLable)
    # dfs(root)





    #Dataset load#
    # attribute_list,AttributeType,X,y=load_breastcancer_data()
    # attribute_list,AttributeType,X,y=load_iris_data()
    attribute_list,AttributeType,X,y=load_textbook_data()


    getCategoricalAttributeValue(attribute_list,X)




    runOnFullDataset(attribute_list,X,y)
    # runCrossValidation(attribute_list,X,y)

    #


