def input_data():
    filename = "Dataset/iris.data"

    attribute_list = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    with open(filename) as f:

        dataPartition=[]
        for line in f:
            # print(line)
            datatuple = {}
            attr_no=0
            classLabel=None
            for x in line.split(','):
                # print(x)
                if attr_no<len(attribute_list):
                    if x.replace('.','',1).isdigit()==True:
                        datatuple[attribute_list[attr_no]]=float(x)

                else:
                    classLabel=x.rstrip()
                attr_no+=1


            # print(datatuple, classLabel)
            dataPartition.append((datatuple,classLabel))


        # for data in dataPartition:
        #     print(data[0]['sepal_length'],data[1])
        return dataPartition


if __name__ == '__main__':
    input_data()