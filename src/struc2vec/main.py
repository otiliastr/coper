from __future__ import absolute_import, division, print_function

from . import graph
from . import struc2vec


# Args is a dictionary here
def exec_struc2vec(args):
	'''
    Pipeline for representational learning for all nodes in a graph.
    '''
	if (args['OPT3']):
		until_layer = args['until_layer']
	else:
		until_layer = None
	print("reading graph")
	G = graph.load_edgelist(args['input'], undirected=True)
	print("constructing struc2vec")
	G = struc2vec.Graph(G, args['directed'], args['workers'], until_layer=until_layer)
	print("done constructing struc2vec")
	print("preprocessing...")
	if (args['OPT1']):
		G.preprocess_neighbors_with_bfs_compact()
	else:
		G.preprocess_neighbors_with_bfs()
	print("preprocessed")
	print("calculating distances...")
	if (args['OPT2']):
		G.create_vectors()
		G.calc_distances(compactDegree=args['OPT1'])
	else:
		G.calc_distances_all_vertices(compactDegree=args['OPT1'])
	print("calculated distances")
	print("creating distances of network...")
	G.create_distances_network()
	print("distances created, preprocessing parameters..")
	G.preprocess_parameters_random_walk()
	print("preprocessed parameters, sumulating walks...")
	G.simulate_walks(args['num_walks'], args['walk_length'])
	print("finished simulating walks, returning...")

	return G
