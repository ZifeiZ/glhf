#!/usr/bin/env python
# coding: utf-8

# #  Power Iteration and Link Preidction

# # 1. Preparation
# Before we start to visualize the networks, we have to install the packages and prepare the network dataset. 
# # 1.1. Install Packages


get_ipython().system(u'pip install matplotlib')
get_ipython().system(u'pip install networkx')
get_ipython().system(u'pip install numpy')



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# # 1.2. Import and Visualize the Graph

get_ipython().magic(u'matplotlib inline')
G = nx.read_edgelist(path="undirected_weighted.edgelist", delimiter=' ', nodetype=int, data=(('weight',float),))
pos = nx.fruchterman_reingold_layout(G)
edges = []
weights = []
for (source, target, weight) in G.edges.data('weight'):
    edges.append((source, target))
    weights.append(weight)
nx.draw_networkx_nodes(G, pos, node_size=200, node_color='orange')
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights*16)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.show()



# # 2. Power Iteration for Eigenvector Centrality

# Get neighbor of node 1： G.neighbors(1)
current_node = 1
list(G.neighbors(current_node))



# Get weights of edges connecting to node 1
for neighbor in G.neighbors(current_node):
    print(G.get_edge_data(neighbor, current_node)['weight'])



Ce = {node:1.0 for node in G.nodes} # create a dictionary with key as the nodes and values are 1
new_score = sum(Ce[neighbor] * G.get_edge_data(neighbor, current_node)['weight'] for neighbor in G.neighbors(current_node))
print(new_score)


# ## 2.2 Implementing Power Iteration


# create a dictionary with key as the nodes and values are 1
Ce = {node:1.0 for node in G.nodes}
maxiter = 100
Ce_record = np.ones((len(G), maxiter+1)) # we use this to record how Ce changes
Ce_tmp = {node:0.0 for node in G.nodes}  # this is used to store the intermediate value of Ce


# main loop
for i in range(0, maxiter):
    
    # for each node, calculate their new eigenvector score and put in Ce_tmp
    for current_node in Ce.keys():
        # aggregate the centrality score from connected neighbors
        Ce_tmp[current_node] = sum(Ce[neighbor] * G.get_edge_data(neighbor, current_node)['weight'] for neighbor in G.neighbors(current_node))

    # normalization
    normalization_term = sum(Ce_tmp[node]**2 for node in Ce_tmp) ** 0.5
    for node in Ce:

        Ce[node] = Ce_tmp[node] / normalization_term

    # record the values
    for node,j in zip(Ce, range(0, len(G))):
        Ce_record[j,i+1] = Ce[node]


# results
print('eigenvector centralities: {}'.format(Ce))

# visualize how the centrality changes
plt.figure()
plt.plot(np.transpose(Ce_record))
plt.show()

# visualiz the graph with node size reflecting the centrality
plt.figure()
nodesize = [Ce[node]*800 for node in Ce]
nx.draw_networkx(G, pos, with_labels=True, node_size=nodesize, font_size=8, node_color=nodesize)
plt.show()


# ## Power Iteration for Katz Centraltiy

# create a dictionary with key as the nodes and values are 1
Ck = {node:1.0 for node in G.nodes}
maxiter = 10
Ck_record = np.ones((len(G), maxiter+1)) # we use this to record how Ck changes
Ck_tmp = {node:0.0 for node in G.nodes}  # this is used to store the intermediate value of Ck
a = 0.85
b = 0.15
# main loop
for i in range(0, maxiter):
    
    # for each node, calculate their new eigenvector score and put in Ck_tmp
    for current_node in Ck.keys():
        # aggregate the centrality score from connected neighbors
        Ck_tmp[current_node] = sum(Ck[neighbor] * G.get_edge_data(neighbor, current_node)['weight'] + b/a for neighbor in G.neighbors(current_node))

    # normalization
    normalization_term = sum(Ck_tmp[node]**2 for node in Ck_tmp) ** 0.5
    for node in Ck:
        Ck[node] = Ck_tmp[node] / normalization_term

    # record the values
    for node,j in zip(Ck, range(0, len(G))):
        Ck_record[j,i+1] = Ck[node]


# results
print('Katz Centralties: {}'.format(Ck))

# visualize how the centrality changes
plt.figure()
plt.plot(np.transpose(Ck_record))
plt.show()

# visualiz the graph with node size reflecting the centrality
plt.figure()
nodesize = [Ck[node]*800 for node in Ck]
nx.draw_networkx(G, pos, with_labels=True, node_size=nodesize, font_size=8, node_color=nodesize)
plt.show()


# ## 2.2 Vectorization

# we can use nx.adjacency_matrix(G) to get the adjacency matrix of G
A = nx.adjacency_matrix(G).todense()
print(A)
plt.matshow(A)



# create a vector of size (A.shape[0], 1)
c = np.ones((A.shape[0], 1))
print(c)


# In[16]:


# update the the centrality score with the equation
c = np.dot(np.transpose(A), c)
print(c)


# In[17]:


# normalization
c = c / np.linalg.norm(c)
print(c)



# initialization
c = np.ones((A.shape[0], 1))

# main loop
maxiter = 10
record_c = np.ones((A.shape[0], maxiter+1))
record_c[:, i] = np.squeeze(c, axis=1)
for i in range(1, maxiter+1):
    c = np.dot(np.transpose(A), c) # c = A^T c
    c = c / np.linalg.norm(c)
    record_c[:,i] = np.squeeze(c, axis=1)

# results
print('eigenvector centralities: {}'.format(c))

# visualize how the centrality changes
plt.figure()
plt.plot(np.transpose(record_c))
plt.show()

# visualiz the graph with node size reflecting the centrality
plt.figure()
nx.draw_networkx(G, pos, with_labels=True, node_size=list(c*800), font_size=8, node_color=nodesize)
plt.show()


# ## Vectorized Version for Katz Centrality


# TODO
all_one_vector = np.ones((A.shape[0], 1))

# initialization
c_k = np.ones((A.shape[0], 1))

# main loop
maxiter = 10
record_c_k = np.ones((A.shape[0], maxiter+1))
record_c_k[:, i] = np.squeeze(c_k, axis=1)
for i in range(1, maxiter+1):
    c_k = a * np.dot(np.transpose(A), c_k) + b * all_one_vector
    c_k = c_k / np.linalg.norm(c_k)
    record_c_k[:,i] = np.squeeze(c_k, axis=1)


# results
print('Katz centralities: {}'.format(c_k))

# visualize how the centrality changes
plt.figure()
plt.plot(np.transpose(record_c_k))
plt.show()

# visualiz the graph with node size reflecting the centrality
plt.figure()
nx.draw_networkx(G, pos, with_labels=True, node_size=list(c*800), font_size=8, node_color=nodesize)
plt.show()


# ##  Eigendecomposition for eigenvector centrality


# eigen decomposition of A
eigenValues, eigenVectors = np.linalg.eigh(A)

# sort the eigenvalues from largest to smallest
idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print(eigenValues)


# In[23]:


# The eigenvector centrality is obtained as the eigenvector corresponding to the largest eigenvalue
c_eig = np.transpose(np.squeeze(eigenVectors[:,0], axis=1))
print(c_eig)


#  Verify that c_eig and c we get in 2.2 are the same


# TODO: Please calcula
print(np.linalg.norm(c - c_eig))



# get neighbors of node u and v
u = 1 # assume u is node 1
v = 2 # assume v is node 2
u_neighbors = set(G.neighbors(u))
v_neighbors = set(G.neighbors(v))
print(u_neighbors)
print(v_neighbors)



# union of two sets
unique_friends = u_neighbors.union(v_neighbors)
print(unique_friends)

# intersect of 
common_friends = u_neighbors.intersection(v_neighbors)
print(common_friends)


# In[27]:


# number of common_friends
len(common_friends)
print(len(common_friends))

# number of unique_friends
len(unique_friends)
print(len(unique_friends))


# ##  Jaccard Similarity of Two nodes

def jaccard_similarity(G, u, v):
    """
    This function calculate the jaccard similarity of two nodes u and v based on the graph structure G
    :param G: the networkx graph
    :param u: node
    :param v: node
    :return: a scalar, the jaccard simialrity of node u and v
    """
    if u not in G.nodes or v not in G.nodes:
        raise ValueError
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    
    similarity = len(u_neighbors.intersection(v_neighbors)) / len(u_neighbors.union(v_neighbors))
    # return the similarity
    return  similarity


# ##  Cosine Similarity

def cosine_similarity(G, u, v):
    """
    This function calculate the cosine similarity of two nodes u and v based on the graph structure G
    :param G: the networkx graph
    :param u: node
    :param v: node
    :return: a scalar, the cosine simialrity of node u and v
    """
    if u not in G.nodes or v not in G.nodes:
        raise ValueError
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    ## plese calculate the cosine simialrity ??????????????????????
    similarity = len(u_neighbors.intersection(v_neighbors)) / (len(u_neighbors) * len(v_neighbors)) ** 0.5
    # return the cosine similarity
    return similarity
    

# ##  Link Prediction 


# TODO: 
user = 1 # the user we want to suggest friends
# the set of nodes that are not linked with the user
preds = []
for node in G.nodes:
    if node not in G.neighbors(user) and node != user:
        # call your function to calculate the jaccard similarity of user and node
        similarity = jaccard_similarity(G,user,node)
        # store the simialrity to preds
        preds.append((user, node, similarity))
        

# rank based on the jaccard similarity
ranked = sorted(preds, key=lambda x: x[2], reverse=True)
print(ranked)
