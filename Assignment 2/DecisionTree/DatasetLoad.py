import pandas



def readDatasetInfo(file):
    filename="Dataset/"+file
    with open(filename) as f:

        attributes=f.readline().rstrip().split(',')
        # classlabels=f.readline().rstrip().split(',')

    return attributes



def load_iris_data():
    attribute_list = readDatasetInfo("irisInfo.txt")

    AttributeType = {}
    for attr in attribute_list:
        AttributeType[attr] = "Continous"
    dataset = pandas.read_csv('Dataset/iris.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y


def load_textbook_data():
    attribute_list = readDatasetInfo("textbookInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    dataset = pandas.read_csv('Dataset/textbook.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y

def load_breastcancer_data():
    attribute_list = readDatasetInfo("breastInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    dataset = pandas.read_csv('Dataset/breast-cancer.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y





def load_wine_data():
    attribute_list = readDatasetInfo("wineInfo.txt")
    dataset = pandas.read_csv('Dataset/wine.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list, X, y
