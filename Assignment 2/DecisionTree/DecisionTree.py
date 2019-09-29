class DecisionTreeNode():

    object_number=0
    def __init__(self, label):
        self.label = label
        self.splitpoint=0
        self.children = []
        self.parent=None
        DecisionTreeNode.object_number += 1
        self.id=DecisionTreeNode.object_number
        status=""
