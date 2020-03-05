#!/usr/bin/env python
# coding: utf-8


get_ipython().system(u'pip install matplotlib')
get_ipython().system(u'pip install networkx')




import networkx as nx
import matplotlib.pyplot as plt


# load the network
undirected_G = nx.read_edgelist("undirected_weighted.edgelist", data=(('weight',float),))


# we can access the node set and edge set with the following functions
print('node list: {}'.format(undirected_G.nodes()))
print('edge list: {}'.format(undirected_G.edges(data = True)))


# Visualization


get_ipython().magic(u'matplotlib inline')
nx.draw(undirected_G) # without layout
plt.show()

pos_random = nx.random_layout(undirected_G) #using random layout
nx.draw(undirected_G, pos_random)
plt.show()

pos_spectral = nx.spectral_layout(undirected_G) #using spectral layout
nx.draw(undirected_G, pos_spectral)
plt.show()

pos_fruchterman = nx.fruchterman_reingold_layout(undirected_G) #using fruchterman reingold layout
nx.draw(undirected_G, pos_fruchterman)
plt.show()



pos_circular = nx.circular_layout(undirected_G)
nx.draw(undirected_G, pos_circular)


# # Usage of the Visualization Tools


# plot the nodes with node_size = 200, node_color as blue
pos_fruchterman = nx.fruchterman_reingold_layout(undirected_G) 
nx.draw_networkx_nodes(undirected_G, pos_fruchterman, node_size=200, node_color='orange')
# plot the edges with width = 2 and edge_color='green'
nx.draw_networkx_edges(undirected_G, pos_fruchterman, width=2, edge_color='green', alpha=1)


nx.draw_networkx_nodes(undirected_G, pos_fruchterman, node_size=200, node_color='orange')
nx.draw_networkx_edges(undirected_G, pos_fruchterman, width=2, edge_color='green', alpha=1)
nx.draw_networkx_labels(undirected_G, pos_fruchterman, font_size=10)




#change the nodes color to yellow
nx.draw_networkx_nodes(undirected_G,pos_fruchterman,node_size=100, node_color='yellow') 
#increase the width of the edges 
nx.draw_networkx_edges(undirected_G,pos_fruchterman,width=1.5,edge_color='black',alpha=1)
#increase the font size
nx.draw_networkx_labels(undirected_G,pos_fruchterman,font_size=12)
#change the color of the selected nodes
nx.draw_networkx_nodes(undirected_G,pos_fruchterman,nodelist=['1','2','3'], node_color='r', node_size=500, alpha=0.8)




nx.draw_networkx_nodes(undirected_G, pos_circular, node_size=200, node_color='green')
nx.draw_networkx_edges(undirected_G, pos_circular, width=2, edge_color= 'grey', alpha=1)
nx.draw_networkx_labels(undirected_G,pos_circular, font_size=15)
nx.draw_networkx_nodes(undirected_G, pos_circular, nodelist=['1','2','3','4'], node_color='r', node_size=600, alpha=0.8)



#change the property of the selected edges
nx.draw_networkx_nodes(undirected_G, pos_fruchterman, node_size=100, node_color='yellow')
nx.draw_networkx_edges(undirected_G, pos_fruchterman,width=1, edge_color='black', alpha=1)
nx.draw_networkx_labels(undirected_G, pos_fruchterman, font_size=10)
nx.draw_networkx_edges(undirected_G, pos_fruchterman,edgelist=[('7','8'),('2','9')], width=8,alpha=0.5, edge_color='r')



nx.draw_networkx_nodes(undirected_G, pos_fruchterman, node_size=100, node_color='yellow')
nx.draw_networkx_edges(undirected_G, pos_fruchterman,width=1, edge_color='black', alpha=1)
nx.draw_networkx_labels(undirected_G, pos_fruchterman, font_size=10)
nx.draw_networkx_edges(undirected_G, pos_fruchterman,edgelist=[('1','9')], width=10,alpha=0.5, edge_color='b')


#  Find Inluential Nodes with Network Centrality and Visualize them


pos = nx.fruchterman_reingold_layout(undirected_G)
nx.draw_networkx(undirected_G,pos,node_size=100,font_size=8)


# Visualize the network with degree centrality



node_degree = nx.degree_centrality(undirected_G)
nodesize=[node_degree[node]*1000 for node in undirected_G.nodes]
nx.draw_networkx(undirected_G, pos, node_size=nodesize, font_size=8, node_color=nodesize)


#  Visualize the network with closeness centrality


closeness_centrality = nx.closeness_centrality(undirected_G)
nodesize = [closeness_centrality[node]*500 for node in undirected_G.nodes]
nx.draw_networkx(undirected_G, pos, with_labels=True, node_size=nodesize, font_size=8, node_color=nodesize)


#   Visualize the network with harmonic centrality



harmonic_centrality = nx.harmonic_centrality(undirected_G)
nodesize = [harmonic_centrality[node]*20 for node in undirected_G.nodes]
nx.draw_networkx(undirected_G, pos, with_labels=True, node_size=nodesize, font_size=8, node_color=nodesize)




#   Visualize the network with katz centrality


katz_centrality = nx.katz_centrality(undirected_G)
nodesize = [katz_centrality[node]*1000 for node in undirected_G.nodes]
nx.draw_networkx(undirected_G, pos, with_labels=True, node_size=nodesize, funt_size=8, node_color=nodesize)


#  Visualize the Shortest Path and Minimum Spanning Tree

#  Shortest Path for Selected Nodes 



#select two nodes 
source = '8'
sink = '16'
#Get the path_node_list from the dijkstra algorithm
length, path_node_list = nx.single_source_dijkstra(undirected_G, source, sink)
print(length, path_node_list)



# convert the node list of the shortest path to edge list
path =[]
for i in range(len(path_node_list)-1):
    path.append((path_node_list[i], path_node_list[i+1]))
print(path)



# now we can visulize the path with by highlighting edges and nodes in the path list
nx.draw_networkx_nodes(undirected_G, pos_fruchterman, node_size=400, node_color='blue')
nx.draw_networkx_edges(undirected_G, pos_fruchterman, width=1, edge_color='gray', alpha=1)
nx.draw_networkx_labels(undirected_G, pos_fruchterman, font_size=10)
nx.draw_networkx_edges(undirected_G, pos_fruchterman, edgelist=path, width=8, alpha=1, edge_color='red')
nx.draw_networkx_nodes(undirected_G, pos_fruchterman, nodelist=path_node_list, node_color='yellow')




# get the minumum spanning tree
T=nx.minimum_spanning_tree(undirected_G)



# The edges of the trees can be retrieved as
print(T.edges)



# visulize the MST
nx.draw_networkx_nodes(undirected_G, pos_fruchterman, node_size=400, node_color='blue')
nx.draw_networkx_edges(undirected_G, pos_fruchterman, width=1, edge_color='gray', alpha=1)
nx.draw_networkx_labels(undirected_G, pos_fruchterman, font_size=10)
nx.draw_networkx_edges(undirected_G, pos_fruchterman, edgelist=T.edges, width=3, alpha=1, edge_color='red')


#select two nodes 
source = '1'
sink = '9'
#Get the path_node_list from the dijkstra algorithm
length, path_node_list = nx.single_source_dijkstra(undirected_G, source, sink)

# convert the node list of the shortest path to edge list
path =[]
for i in range(len(path_node_list)-1):
    path.append((path_node_list[i], path_node_list[i+1]))

# now we can visulize the path with by highlighting edges and nodes in the path list
nx.draw_networkx_nodes(undirected_G, pos_fruchterman, node_size=400, node_color='blue')
nx.draw_networkx_edges(undirected_G, pos_fruchterman, width=1, edge_color='gray', alpha=1)
nx.draw_networkx_labels(undirected_G, pos_fruchterman, font_size=10)
nx.draw_networkx_edges(undirected_G, pos_fruchterman, edgelist=path, width=8, alpha=1, edge_color='red')
nx.draw_networkx_nodes(undirected_G, pos_fruchterman, nodelist=path_node_list, node_color='yellow')

