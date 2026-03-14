import numpy as np
import pickle
import os
import sys

#ACGCCTGACACGCGCT-1,0,0,0,558,543
#TACCGATCCAACACTT-1,0,1,1,601,568
#ATTAAAGCGGACGAGC-1,0,0,2,558,593
def read_coord(n, good_cells):
	Xcen = []
	cells = []
	with open(n) as f:
		for l in f:
			l = l.rstrip("\n")
			ll = l.split(",")
			cells.append(ll[0])
			Xcen.append((float(ll[-2]), float(ll[-1])))

	map_cell = {}
	for ic,c in enumerate(cells):
		map_cell[c] = ic

	good_cell_ids = np.array([map_cell[c] for c in good_cells])
	cells = np.array(cells)
	
	tmp_Xcen = np.empty((len(Xcen), 2), dtype="float32")
	for ind, (i,j) in enumerate(Xcen):
		tmp_Xcen[ind,:] = [i,j]
	
	cells = cells[good_cell_ids]
	Xcen = tmp_Xcen[good_cell_ids, :]

	Xcen2 = np.empty((len(Xcen), 2), dtype="float32")
	for ind, (i, j) in enumerate(Xcen):
		Xcen2[ind, :] = [j, -1.0 * i]
	Xcen = Xcen2
	return Xcen, cells

def read_expression(n):
	with open(n) as f:
		h = f.readline().rstrip("\n").split()
		h = [xh.replace(".", "-") for xh in h]
		num_gene = sum(1 for _ in f)
	
	mat = np.empty((num_gene, len(h)), dtype="float32")
	genes = []
	
	with open(n) as f:
		f.readline()  # skip header
		for ig, l in enumerate(f):
			l = l.rstrip("\n")
			ll = l.split()
			gene = ll[0]
			values = [float(v) for v in ll[1:]]
			mat[ig, :] = values
			genes.append(gene)
	
	return mat, h, genes

def save_dataset(dset):
	#ACGCCTGACACGCGCT-1,0,0,0,558,543
	#TACCGATCCAACACTT-1,0,1,1,601,568
	#ATTAAAGCGGACGAGC-1,0,0,2,558,593
	print(f"Loading and saving data for dataset: {dset}")
	#mat_file = "%s/Giotto_norm_expr.txt" % dset
	mat = np.load(dset, allow_pickle=True)

	scale = 6000.0
	mat = mat.astype(np.float32, copy=False)
	col_sums = mat.sum(axis=0, keepdims=True)
	mat_norm = mat / col_sums * scale
	mat = np.log1p(mat)


	#GSM5808054_new_genes.npy
	dset_base = dset.split("_")[0]
	genes = np.load(dset_base + "_new_genes.npy", allow_pickle=True)
	cells = np.load(dset_base + "_new_spots.npy", allow_pickle=True)
	coord_file = dset_base + "_tissue_positions_list.csv"	
	# Ensure the files exist before processing
	'''
	if not os.path.exists(mat_file) or not os.path.exists(coord_file):
		print(f"Files for dataset {dset} do not exist. Skipping.")
		return
	'''
	print(cells.shape)
	#mat, cells, genes = read_expression(mat_file)
	Xcen, Xcells = read_coord(coord_file, cells)
	print(Xcen.shape)
	print(Xcells.shape)
	dataset = {
		'mat': mat,
		'cells': cells,
		'genes': genes,
		'Xcen': Xcen,
		'Xcells': Xcells
	}
	with open(f'{dset_base}_data.pkl', 'wb') as f:
		pickle.dump(dataset, f)
	print(f"Dataset {dset} has been saved in '{dset_base}_data.pkl'.")

def load_all_datasets(gsm_list_file):
	dset_list = []
	coord_list = []
	with open(gsm_list_file) as f:
		#dset_list = [line.strip().split("\t")[0] for line in f]
		for line in f:
			line = line.strip().split("\t")
			dset_list.append(line[0])
			#coord_list.append(line[1])

	for dset in dset_list:
		save_dataset(dset)

if __name__ == '__main__':
	#load_all_datasets('ravi.list')
	load_all_datasets(sys.argv[1])


''' 
def read_coord(n):
	f = open(n)
	Xcen = []
	cells = []
	for l in f:
		l = l.rstrip("\n")
		ll = l.split(",")
		cells.append(ll[0])
		Xcen.append((float(ll[-2]), float(ll[-1])))
	f.close()
	Xcen2 = np.empty((len(Xcen), 2), dtype="float32")
	for ind,(i,j) in enumerate(Xcen):
		Xcen2[ind, :] = [j,-1.0*i]
	Xcen = Xcen2
	return Xcen, cells

def read_expression(n):
	f = open(n)
	h = f.readline().rstrip("\n").split()
	h = [xh.replace(".", "-") for xh in h]
	num_gene = 0
	for l in f:
		l = l.rstrip("\n")
		num_gene+=1
	f.close()
	mat = np.empty((num_gene, len(h)), dtype="float32")
	f = open(n)
	f.readline()
	ig = 0
	genes = []
	for l in f:
		l = l.rstrip("\n")
		ll = l.split()
		gene = ll[0]
		values = [float(v) for v in ll[1:]]
		mat[ig,:] = values
		genes.append(gene)
		ig+=1
	f.close()
	return mat, h, genes

def read_dataset(dset):
	def actual_reading(dset):
		t =  time.process_time()
		print(f"Reading dataset {dset}")
		sys.stdout.flush()
		mat, cells, genes = read_expression(f"dir_{dset}/Giotto_norm_expr.txt")
		Xcen, Xcells = read_coord(f"dir_{dset}/spatial/tissue_positions_list.csv")
		#end_time = time.time()
		print(f"Reading {dset} took {t: 2f} seconds")
		sys.stdout.flush()
		return dset, mat, cells, genes, Xcen, Xcells
	return delayed(actual_reading)(dset)

def read_all_datasets(n):
	with open(n) as f:
		dset_list = [line.rstrip("\n") for line in f]

	tasks = [read_dataset(dset) for dset in dset_list]
	results = compute(*tasks)

	all_mat, all_cells, all_genes, all_Xcen, all_Xcells = {}, {}, {}, {}, {}
	for dset, mat, cells, genes, Xcen, Xcells in results:
		all_mat[dset] = mat
		all_cells[dset] = cells
		all_genes[dset] = genes
		all_Xcen[dset] = Xcen
		all_Xcells[dset] = Xcells

	return all_mat, all_cells, all_genes, all_Xcen, all_Xcells
'''
