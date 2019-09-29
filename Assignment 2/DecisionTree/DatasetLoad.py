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

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Continous"
    dataset = pandas.read_csv('Dataset/wine.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]


    return attribute_list,AttributeType,X,y

def load_tictactoe_data():
    attribute_list = readDatasetInfo("tic-tac-toe.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    # dataset = pandas.read_csv('Dataset/tic-tac-toe.data')
    dataset = pandas.read_csv('Dataset/tic-tac-toe.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]


    return attribute_list,AttributeType,X,y


def load_balancescale_data():
    attribute_list = readDatasetInfo("balance-scaleInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    dataset = pandas.read_csv('Dataset/balance-scale.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y


def load_agaricus_data():
    attribute_list = readDatasetInfo("agaricus-lepiotaInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    dataset = pandas.read_csv('Dataset/agaricus-lepiota.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y


def load_nursery_data():
    attribute_list = readDatasetInfo("nurseryInfo.txt")

    AttributeType = {}
    for attr in attribute_list:
        AttributeType[attr] = "Categorical"
    dataset = pandas.read_csv('Dataset/nursery.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y

def load_abalone_data():
    attribute_list = readDatasetInfo("abaloneInfo.txt")



    AttributeType={'Sex':'Categorical','Length':'Continous','Diameter':'Continous','Height':'Continous','Whole weight':'Continous','Shucked weight':'Continous','Viscera weight':'Continous','Shell weight':'Continous','Rings':'Continous'}


    dataset = pandas.read_csv('Dataset/abalone.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y

def load_flare_data():
    attribute_list = readDatasetInfo("flareInfo.txt")

    AttributeType = {}
    for attr in attribute_list:
        AttributeType[attr] = "Categorical"
    dataset = pandas.read_csv('Dataset/flare.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y


# copy paste dataset, create datasetInfo, fix them. load_dateset() method,  change filesnames
# attribute types, # call the method from main