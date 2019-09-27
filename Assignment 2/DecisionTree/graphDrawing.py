import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import networkx.generators.small as gs
import random

def draw_graph(G,edgeLabels):

# G = nx.DiGraph()


# figure(num=None, figsize=(4, 4), dpi=80, facecolor='w', edgecolor='k')

# write dot file to use with graphviz
    # run "dot -Tpng test.dot >test.png"
    write_dot(G,'test.dot')

    # same layout using matplotlib with no labels
    plt.title('draw_networkx')
    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)

    nx.draw_networkx_edge_labels(G,pos,edge_labels=edgeLabels,font_color='red')
    # for v in G.node():
    #     print(len(v))


    # nx.draw(G, with_labels=True,  arrows=True)
    # nx.draw_networkx_nodes(G, pos, node_size=[len(v) * 100 for v in G.nodes()])

    plt.show()


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edge(1,2)
    edgeLabel=[]
    # edgeLabel.append(((1,2),"xxx"))
    draw_graph(G,edgeLabel)