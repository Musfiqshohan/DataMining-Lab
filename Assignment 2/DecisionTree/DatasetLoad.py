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
    datasetname="iris"
    dataset = pandas.read_csv('Dataset/iris.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname


def load_textbook_data():
    attribute_list = readDatasetInfo("textbookInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    datasetname="textbook"
    dataset = pandas.read_csv('Dataset/textbook.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname

def load_breastcancer_data():
    attribute_list = readDatasetInfo("breastInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    datasetname="breast-cancer"
    dataset = pandas.read_csv('Dataset/breast-cancer.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname


def load_wine_data():
    attribute_list = readDatasetInfo("wineInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Continous"
    datasetname="wine"
    dataset = pandas.read_csv('Dataset/wine.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]


    return attribute_list,AttributeType,X,y,datasetname

def load_tictactoe_data():
    attribute_list = readDatasetInfo("tic-tac-toe.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    # datasetname=""dataset = pandas.read_csv('Dataset/tic-tac-toe.data')
    datasetname="tic-tac-toe"
    dataset = pandas.read_csv('Dataset/tic-tac-toe.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]


    return attribute_list,AttributeType,X,y,datasetname


def load_balancescale_data():
    attribute_list = readDatasetInfo("balance-scaleInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    datasetname="balance-scale"
    dataset = pandas.read_csv('Dataset/balance-scale.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname


def load_agaricus_data():
    attribute_list = readDatasetInfo("agaricus-lepiotaInfo.txt")

    AttributeType={}
    for attr in attribute_list:
        AttributeType[attr]="Categorical"
    datasetname="agaricus-lepiota"
    dataset = pandas.read_csv('Dataset/agaricus-lepiota.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup)
    colList = [x for x in range(1,dimension)]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname


def load_nursery_data():
    attribute_list = readDatasetInfo("nurseryInfo.txt")

    AttributeType = {}
    for attr in attribute_list:
        AttributeType[attr] = "Categorical"
    datasetname="nursery"
    dataset = pandas.read_csv('Dataset/nursery.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname

def load_abalone_data():
    attribute_list = readDatasetInfo("abaloneInfo.txt")



    AttributeType={'Sex':'Categorical','Length':'Continous','Diameter':'Continous','Height':'Continous','Whole weight':'Continous','Shucked weight':'Continous','Viscera weight':'Continous','Shell weight':'Continous','Rings':'Continous'}


    datasetname="abalonetest"
    dataset = pandas.read_csv('Dataset/abalone.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname

def load_flare_data():
    attribute_list = readDatasetInfo("flareInfo.txt")

    AttributeType = {}
    for attr in attribute_list:
        AttributeType[attr] = "Categorical"
    datasetname="flare"
    dataset = pandas.read_csv('Dataset/flare.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname


def load_banknote_data():
    attribute_list = readDatasetInfo("data_banknote_authenticationInfo.txt")

    AttributeType = {}
    for attr in attribute_list:
        AttributeType[attr] = "Continous"

    datasetname="data_banknote_authentication"
    dataset = pandas.read_csv('Dataset/data_banknote_authentication.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname



def load_band_data():
    attribute_list = readDatasetInfo("bandsInfo.txt")

    AttributeType = {}

    continuatts = ['timestamp', 'proof cut', 'viscosity', 'caliper', 'ink temperature', 'humifity', 'roughness',
                   'blade pressure', 'varnish pct', 'press speed', 'ink pct', 'solvent pct', 'ESA Voltage',
                   'ESA Amperage', 'wax', 'hardener', 'roller durometer', 'current density', 'anode space ratio',
                   'chrome content']
    cateattr = ['cylinder number', 'customer', 'job number', 'grain screened', 'ink color', 'proof on ctd ink',
                'blade mfg', 'cylinder division', 'paper type', 'ink type', 'direct steam', 'solvent type',
                'type on cylinder', 'press type', 'press', 'unit number', 'cylinder size', 'paper mill location',
                'plating tank']


    for attr in continuatts:
        AttributeType[attr]="Continous"
    for attr in cateattr:
        AttributeType[attr]="Categorical"


    datasetname="Cylinder Bands"
    dataset = pandas.read_csv('Dataset/bands.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [x for x in range(dimension)]
    classlabList = [dimension]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname

def load_wdbc_data():
    attribute_list = readDatasetInfo("wdbcInfo.txt")

    AttributeType = {}
    for attr in attribute_list:
        AttributeType[attr] = "Continous"

    datasetname="wdbc"
    dataset = pandas.read_csv('Dataset/wdbc.data')

    singletup = dataset.iloc[1, :]
    dimension = len(singletup) - 1
    colList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    classlabList = [0]
    X = dataset.iloc[:, colList]
    y = dataset.iloc[:, classlabList]

    return attribute_list,AttributeType,X,y,datasetname

# copy paste dataset, create datasetInfo, fix them. load_dateset() method,  change filesnames
# attribute types, # call the method from main