import pickle
import sys
import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean,squareform,pdist

def get_adjacency_list(dist=None, cutoff=None):
	ncell = dist.shape[0]
	points = set([])
	edges = set([])
	adjacent = {}
	for i in range(ncell):
		for j in range(i+1, ncell):
			if dist[i,j]<=cutoff:
				edges.add(tuple(sorted([i, j])))
				points = points | set([i,j])
	for i in range(ncell):
		if i in points: continue
		dist_i = sorted([(dist[i,j], j) for j in range(ncell) if i!=j])
		edges.add(tuple(sorted([i, dist_i[0][1]])))
	for e1, e2 in edges:
		adjacent.setdefault(e1, set([]))
		adjacent.setdefault(e2, set([]))
		adjacent[e1].add(e2)
		adjacent[e2].add(e1)
	return edges, adjacent
	
def test_adjacency_list(Xcen, percentages, metric="euclidean"):
	output = []
	percentile = {}
	dist = pdist(Xcen, metric=metric)
	s_dist = squareform(dist)
	for px in percentages:
		percentile[px] = np.percentile(dist, px)
		edges, adjacent = get_adjacency_list(dist=s_dist, cutoff=percentile[px])
		avg_neighbor = np.mean([len(adjacent[n]) for n in adjacent])
		print("cutoff:%.2f%%" % px, "#nodes:%d" % len(adjacent), \
		"#edges:%d"%len(edges), "avg.nei:%.2f" % avg_neighbor)
		output.append({"cutoff":px, "nodes":len(adjacent), "edges":len(edges), "avg.nei":avg_neighbor})
	return output

def calc_neighbor_graph(Xcen, cutoff, metric="euclidean"):
	dist = pdist(Xcen, metric=metric)
	s_dist = squareform(dist)
	cutoff = cutoff
	percent_value = np.percentile(dist, cutoff)
	edges, adjacent = get_adjacency_list(dist=s_dist, cutoff=percent_value)
	return edges, adjacent

def build_delaunay(points):
	tri = Delaunay(points)
	graph = nx.Graph()
	# Add nodes to the graph, using the index of the point in the array as the node ID
	for i in range(points.shape[0]):
		graph.add_node(i)
	# Add edges based on Delaunay triangulation
	for simplex in tri.simplices:
		for i in range(len(simplex)):
			for j in range(i + 1, len(simplex)):
				graph.add_edge(simplex[i], simplex[j])
	return graph

if __name__=="__main__":
	#file_path = f'{dset}_data.pkl'
	file_path = sys.argv[1]
	data = None
	with open(file_path, 'rb') as f:
		data = pickle.load(f)
	
	mat = data["mat"]
	cells = data["cells"]
	genes = data["genes"]

	Xcen = data["Xcen"]
	Xcells = data["Xcells"]

	map_cell = {}
	for ic,c in enumerate(Xcells):
		map_cell[c] = ic

	cell_ids = np.array([map_cell[c] for c in cells])
	Xcen = Xcen[cell_ids, :]
	Xcells = np.array(Xcells)[cell_ids]

	print(mat.shape)
	print(Xcen.shape)	

	delaunay_graph = build_delaunay(Xcen)

	good_per = []
	x = 0.005
	while True:
		output = test_adjacency_list(Xcen, [x], metric="euclidean")
		output = output[0]
		if output["avg.nei"]>=6 and output["avg.nei"]<=9:
			good_per.append((x, output["avg.nei"]))
			break	
		if output["avg.nei"]>9:
			break
		x+=0.005

	final_x = good_per[0][0]
	edges, adjacent = calc_neighbor_graph(Xcen, final_x, metric="euclidean")

	data["Xcen"] = Xcen
	data["Xcells"] = Xcells
	data["graph_cutoff"] = final_x
	data["edges"] = edges
	data["adjacent"] = adjacent

	mat_nei = np.empty(mat.shape, dtype="float32")
	for i in range(Xcen.shape[0]):
		nei_ids = np.array(list(adjacent[i]))
		#print(i, nei_ids)
		mat_nei[:,i] = np.mean(mat[:,nei_ids], axis=1)

	data["Xnei"] = mat_nei

	fname = sys.argv[1]
	fname = fname.replace("_data.pkl", "_with_nei.pkl")
	with open(fname, 'wb') as f:
		pickle.dump(data, f)
	#print(edges)

	#pos = {i: Xcen[i] for i in range(Xcen.shape[0])} # Node positions for plotting
	#nx.draw(delaunay_graph, pos, with_labels=False, node_color='skyblue', node_size=50, edge_color='gray')

	
	#plt.title("Delaunay Neighbor Graph")

	#plt.scatter(Xcen[:,0], Xcen[:,1])
	#plt.show()
