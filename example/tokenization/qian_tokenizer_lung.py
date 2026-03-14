"""
Tokenizer for ST Visium Datasets converted into Anndata Files modified and using source material from 
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .h5ad anndata file
Required row (gene) attribute: "ensembl_id"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:

  tk = ST_TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("anndata_directory", "output_directory", "output_prefix")
"""
from __future__ import annotations
from typing import Literal
import pickle
from pathlib import Path

import logging

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import anndata as ad
import loompy as lp
import numpy as np
import scipy.sparse as sp
from datasets import Dataset
from datasets import concatenate_datasets
from sklearn.model_selection import train_test_split
import pyarrow as pa
from typing import List,Optional

logger = logging.getLogger(__name__)

GENE_MEDIAN_FILE = Path(__file__).parent / "detected_gene_median_dict.pickle"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "new_token_dictionary.pickle"

def read_cell_metadata(fname, cells):
	f = open(fname)
	h = f.readline().rstrip("\n").split(",")[1:]
	keys = {}
	for k in h:
		keys.setdefault(k, [])
	keys["name"] = []
	for l in f:
		l = l.rstrip("\n")
		ll = l.split(",")
		cell_id = ll[0].strip("\"")
		keys["name"].append(cell_id)
		for ih,ic in zip(h, ll[1:]):
			keys[ih].append(ic.strip("\""))
	f.close()
	remain = [c for c in cells if c not in set(keys["name"])]
	for ik in keys:
		if ik!="name":
			for t_item in remain:
				keys[ik].append(keys[ik][0])
	for t_item in remain:
		keys["name"].append(t_item)
	for ik in keys:
		keys[ik] = np.array(keys[ik])

	map_cell = {}
	for ic,c in enumerate(keys["name"]):
		map_cell[c] = ic
	
	good_cell_ids = []
	for c in cells:
		good_cell_ids.append(map_cell[c])
	good_cell_ids = np.array(good_cell_ids)

	for ik in keys:
		if ik!="name":
			keys[ik] = keys[ik][good_cell_ids]
	keys["name"] = keys["name"][good_cell_ids]
	del keys["name"]
	return keys



def check_anndata_format(adata):
	"""
	Check the format of an AnnData object.

	Parameters:
	- adata: An AnnData object.

	Returns:
	- result: A dictionary containing the results of the checks.
	"""

	result = {"Anndata has Valid Format": True, "messages": []}
	result["Dimension"] = adata.X.shape
	result["Dtype"] = adata.X.dtype

	# Check if 'ensembl_id' is present in adata.var
	if "ensembl_id" not in adata.var:
		result["Anndata has Valid Format"] = False
		result["messages"].append("Error: 'ensembl_id' is not present in adata.var.")

	# Check if 'n_counts' is present in adata.obs
	if "n_counts" not in adata.obs:
		result["Anndata has Valid Format"] = False
		result["messages"].append("Error: 'n_counts' is not present in adata.obs.")

	# Check if adata.X contains only integer values
	if len(adata.X.shape) > 1 and not np.issubdtype(adata.X.dtype, np.number):
		result["Anndata has Valid Format"] = False
		result["messages"].append(f"Error: adata.X contains non-numeric values of type {adata.X.dtype}.")

	return result


def generate_dict_summary(data_dict):
	# Check if the dictionary is empty
	if not data_dict:
		print("Dictionary is empty.")
		return

	key = next(iter(data_dict.keys()))
	value = data_dict[key]

	print(f"Summary of the dictionary:")
	print(f"Key: {key}")

	# Check if the value is a NumPy array or a list of NumPy arrays
	if isinstance(value, np.ndarray):
		print(f"Type: NumPy array")
		print(f"Dtype: {value.dtype}")
		print(f"Shape: {value.shape}")
	elif isinstance(value, list) and all(isinstance(arr, np.ndarray) for arr in value):
		print(f"Type: List of NumPy arrays")
		print(f"Dtypes: {np.unique([arr.dtype for arr in value])}")
		shapes = [arr.shape for arr in value]
		print(f"Range of shapes: {min(shapes)} to {max(shapes)}")
	else:
		print(f"Type: {type(value)}")


def rank_genes(gene_vector, gene_tokens):
	"""
	Rank gene expression vector.
	"""
	# sort by median-scaled gene values
	sorted_indices = np.argsort(-gene_vector)
	return gene_tokens[sorted_indices]


def tokenize_cell(gene_vector, gene_tokens):
	"""
	Convert normalized gene expression vector to tokenized rank value encoding.
	"""
	# create array of gene vector with token indices
	# mask undetected genes
	nonzero_mask = np.nonzero(gene_vector)[0]
	# rank by median-scaled gene values
	return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


class ST_TranscriptomeTokenizer:
	def __init__(
		self,
		custom_attr_name_dict=None,
		nproc=1,
		downsample_percent = None,
		downsample_seed = None,
		gene_median_file=GENE_MEDIAN_FILE,
		token_dictionary_file=TOKEN_DICTIONARY_FILE,
	):
		"""
		Initialize tokenizer.

		Parameters
		----------
		custom_attr_name_dict : None, dict
			Dictionary of custom attributes to be added to the dataset.
			Keys are the names of the attributes in the loom file.
			Values are the names of the attributes in the dataset.
		nproc : int
			Number of processes to use for dataset mapping.
		gene_median_file : Path
			Path to pickle file containing dictionary of non-zero median
			gene expression values across Genecorpus-30M.
		token_dictionary_file : Path
			Path to pickle file containing token dictionary (Ensembl IDs:token).
		"""
		# dictionary of custom attributes {output dataset column name: input .loom column name}
		self.custom_attr_name_dict = custom_attr_name_dict

		# number of processes for dataset mapping
		self.nproc = nproc

		# load dictionary of gene normalization factors
		# (non-zero median value of expression across Genecorpus-30M)
		with open(gene_median_file, "rb") as f:
			self.gene_median_dict = pickle.load(f)

		# load token dictionary (Ensembl IDs:token)
		with open(token_dictionary_file, "rb") as f:
			self.gene_token_dict = pickle.load(f)

		# gene keys for full vocabulary
		self.gene_keys = list(self.gene_median_dict.keys())

		# protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
		self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))
		
		# for making downsampled dataset representations
		self.downsample_percent = downsample_percent
		self.downsample_seed = downsample_seed

	def tokenize_data(
		self,
		data_directory: Path | str,
		output_directory: Path | str,
		output_prefix: str,
		use_generator: bool = False,
	):
		"""
		Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.

		Parameters
		----------
		loom_data_directory : Path
			Path to directory containing loom files or anndata files
		output_directory : Path
			Path to directory where tokenized data will be saved as .dataset
		output_prefix : str
			Prefix for output .dataset
		use_generator : bool
			Whether to use generator or dict for tokenization.
		"""
		tokenized_cells, tokenized_neighbors, cell_metadata = self.tokenize_files(Path(data_directory))
		tokenized_dataset = self.create_dataset(tokenized_cells, tokenized_neighbors, cell_metadata, use_generator=use_generator)

		output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
		tokenized_dataset.save_to_disk(output_path)

	def tokenize_files(self, data_directory):
		tokenized_cells = []
		tokenized_neighbors = []
		if self.custom_attr_name_dict is not None:
			cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
			cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

		# loops through directories to tokenize .loom files
		file_found = 0
		# loops through directories to tokenize .h5ad files
		tokenize_file_fn = (self.tokenize_pickle)
		for file_path in data_directory.glob("*.pkl"):
			file_found = 1
			print(f"Tokenizing {file_path}")
			file_tokenized_cells, file_tokenized_neighbors, file_cell_metadata = tokenize_file_fn(file_path,downsample_percent=self.downsample_percent,downsample_seed=self.downsample_seed)
			tokenized_cells += file_tokenized_cells
			tokenized_neighbors += file_tokenized_neighbors

			if self.custom_attr_name_dict is not None:
				for k in cell_attr:
					cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
			else:
				cell_metadata = None

		if file_found == 0:
			logger.error(
				f"No .h5ad files found in directory {data_directory}.")
			raise
		return tokenized_cells, tokenized_neighbors, cell_metadata

	def tokenize_pickle(self, file_path, chunk_size=512, downsample_percent=None, downsample_seed=None):
		data = None
		with open(file_path, 'rb') as f:
			data = pickle.load(f)
		mat = data["mat"]
		cells = data["cells"]
		genes = data["genes"]
		Xcen = data["Xcen"]
		Xcells = data["Xcells"]
		Xnei = data["Xnei"]

		print(file_path)
		filebase = str(file_path).split("/")[-1]
		csv_file = filebase.replace(".h5ad_with_nei.pkl", ".csv").replace("ind_", "")
		metadata = read_cell_metadata("col.metadata/%s" % csv_file, cells)

		print("mat shape", mat.shape)
		print("cells shape", len(cells))
		print("Xcen shape", Xcen.shape)
		print("Xcells shape", Xcells.shape)
		print("Xnei shape", Xnei.shape)
		print("metadata shape", metadata["study"].shape)

		metadata["barcode"] = np.array(cells)
		metadata["coord_x"] = np.array(Xcen[:, 0])
		metadata["coord_y"] = np.array(Xcen[:, 1])

		print("coord_x shape", metadata["coord_x"].shape)
		print("coord_y shape", metadata["coord_y"].shape)

		##### DOWN SAMPLE CODE #####
		if downsample_percent is not None:
			percentage_to_select = downsample_percent
			print(f"Downsampling Study Observations by {(downsample_percent) *100} % ")
			selected_indices, _ = train_test_split(range(len(cells)), test_size=percentage_to_select, random_state=downsample_seed)
			mat = mat[:, selected_indices]
			cells = np.array(cells)[selected_indices].tolist()
			Xcen = Xcen[selected_indices,:]
			Xcells = cells
			Xnei = Xnei[:, selected_indices]
		#############################

		if self.custom_attr_name_dict is not None:
			file_cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

		genes = np.array(genes)
		coding_miRNA_loc = np.where([self.genelist_dict.get(i, False) for i in genes])[0]
		norm_factor_vector = np.array([self.gene_median_dict[i] for i in genes[coding_miRNA_loc]])
		coding_miRNA_ids = genes[coding_miRNA_loc]
		coding_miRNA_tokens = np.array([self.gene_token_dict[i] for i in coding_miRNA_ids])

		var_exists = False #for checking if "filter_pass" exists in adata.obs["filter_pass"]

		if var_exists:
			#tokenize filtered cells
			filter_pass_loc = np.where([i == 1 for i in data["filter_pass"]])[0]
		else:
			#tokenize all cells
			filter_pass_loc = np.array([i for i in range(mat.shape[1])])

		tokenized_cells = []
		tokenized_neighbors = []
		for i in range(0, len(filter_pass_loc), chunk_size):
			idx = filter_pass_loc[i:i+chunk_size]
			X_view = mat[:, idx]
			X_view = X_view[coding_miRNA_loc, :]
			X_norm = X_view - norm_factor_vector[:, np.newaxis]
			#print(X_norm)
			#print(norm_factor_vector)
			X_norm = np.transpose(X_norm)
			X_norm = sp.csr_matrix(X_norm)
			tokenized_cells += [rank_genes(X_norm[xi].data, coding_miRNA_tokens[X_norm[xi].indices]) for xi in range(X_norm.shape[0])]
			# add custom attributes for subview to dict
			if self.custom_attr_name_dict is not None:
				for k in file_cell_metadata.keys():
					#==================================
					file_cell_metadata[k] += metadata[k][idx].tolist()
					#file_cell_metadata[k] += adata[idx].obs[k].tolist() #still implement!!!!!
					#==================================
			else:
				file_cell_metadata = None

			X_view2 = Xnei[:, idx]
			X_view2 = X_view2[coding_miRNA_loc,:]
			X_norm2 = X_view2 - norm_factor_vector[:, np.newaxis]
			X_norm2 = np.transpose(X_norm2) # now it is cell by gene matrix
			X_norm2 = sp.csr_matrix(X_norm2)
			tokenized_neighbors += [rank_genes(X_norm2[xi].data, coding_miRNA_tokens[X_norm2[xi].indices]) for xi in range(X_norm2.shape[0])]
		
		print("tokenized_cell size", len(tokenized_cells))
		print("tokenized_neighbor size", len(tokenized_neighbors))
		print("study", len(file_cell_metadata["study"]))
	
		return tokenized_cells, tokenized_neighbors, file_cell_metadata

	def tokenize_anndata(self, adata_file_path, target_sum=10_000, chunk_size=512,downsample_percent=None,downsample_seed=None):
		#adata = ad.read(adata_file_path, backed="r")
		adata = ad.read(adata_file_path)
		print(adata.obs["n_counts"])
		adata = adata[adata.obs['n_counts'] != 0]
		print(adata)	
		
		##### DOWN SAMPLE CODE #####
		if downsample_percent is not None:
			percentage_to_select = downsample_percent
			print(f"Downsampling Study Observations by {(downsample_percent) *100} % ")
			selected_indices, _ = train_test_split(range(len(adata.obs)), test_size=percentage_to_select, random_state=downsample_seed)
			adata = adata[selected_indices, :]
		#############################
		
		check_format = check_anndata_format(adata)
		print(check_format)

		if self.custom_attr_name_dict is not None:
			file_cell_metadata = {
				attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
			}

		coding_miRNA_loc = np.where(
			[self.genelist_dict.get(i, False) for i in adata.var["ensembl_id"]]
		)[0]
		norm_factor_vector = np.array(
			[
				self.gene_median_dict[i]
				for i in adata.var["ensembl_id"][coding_miRNA_loc]
			]
		)
		coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_loc]
		coding_miRNA_tokens = np.array(
			[self.gene_token_dict[i] for i in coding_miRNA_ids]
		)

		try:
			_ = adata.obs["filter_pass"]
		except KeyError:
			var_exists = False
		else:
			var_exists = True

		if var_exists:
			filter_pass_loc = np.where(
				[i == 1 for i in adata.obs["filter_pass"]]
			)[0]
		elif not var_exists:
# =============================================================================
#			 print(
#				 f"{adata_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
#			 )
# =============================================================================
			filter_pass_loc = np.array([i for i in range(adata.shape[0])])

		tokenized_cells = []

		for i in range(0, len(filter_pass_loc), chunk_size):
			idx = filter_pass_loc[i:i+chunk_size]

			n_counts = adata[idx].obs['n_counts'].values[:, None]
			X_view = adata[idx, coding_miRNA_loc].X
			#X_norm = (X_view / n_counts * target_sum / norm_factor_vector)
			X_norm = (X_view - norm_factor_vector)
			print(X_norm)
			print(norm_factor_vector)
			X_norm = sp.csr_matrix(X_norm)

			tokenized_cells += [
				rank_genes(X_norm[xi].data, coding_miRNA_tokens[X_norm[xi].indices])
				for xi in range(X_norm.shape[0])
			]

			# add custom attributes for subview to dict
			if self.custom_attr_name_dict is not None:
				for k in file_cell_metadata.keys():
					file_cell_metadata[k] += adata[idx].obs[k].tolist()
			else:
				file_cell_metadata = None

		return tokenized_cells, file_cell_metadata

	#batch_size originally 10000
	def create_dataset(self, tokenized_cells, tokenized_neighbors, cell_metadata, use_generator=False, batch_size = 1000):
		print("Creating dataset.")
		# create dict for dataset creation
		dataset_dict = {"input_ids": tokenized_cells, "neighbor_ids": tokenized_neighbors}


		print("dataset dict size", len(dataset_dict["input_ids"]))
		if self.custom_attr_name_dict is not None:
			dataset_dict.update(cell_metadata)

		generate_dict_summary(dataset_dict)
		print(dataset_dict.keys())
		# create dataset
		if use_generator:
			def dict_generator():
				for i in range(len(tokenized_cells)):
					yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
					
			keys, values = zip(*dict_generator.items())

			# Initialize an empty list to store batches
			batches = []
		
			# Process the dictionary in batches
			for i in range(0, len(values[0]), batch_size):
				batch_dict = {key: value[i:i+batch_size] for key, value in zip(keys, values)}
				batches.append(Dataset.from_dict(batch_dict))
		
			# Concatenate the batches into one large dataset
			print("Concatenating Datasets")
			print(batches)
			output_dataset = concatenate_datasets(batches)

			#output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
		else:
			#output_dataset = Dataset.from_dict(dataset_dict)
			keys, values = zip(*dataset_dict.items())

			# Initialize an empty list to store batches
			batches = []
			# Process the dictionary in batches
			for i in range(0, len(values[0]), batch_size):
				batch_dict = {key: value[i:i+batch_size] for key, value in zip(keys, values)}
				batches.append(Dataset.from_dict(batch_dict))
			# Concatenate the batches into one large dataset
			print("Concatenating Datasets")
			print(batches)
			output_dataset = concatenate_datasets(batches)
			

		# truncate dataset
		def truncate(example):
			example["input_ids"] = example["input_ids"][:2048]
			example["neighbor_ids"] = example["neighbor_ids"][:2048]
			example["input_ids"] = example["input_ids"] + example["neighbor_ids"]
			del example["neighbor_ids"]
			return example

		output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)

		# measure lengths of dataset
		def measure_length(example):
			example["length"] = len(example["input_ids"])
			return example

		output_dataset_truncated_w_length = output_dataset_truncated.map(measure_length, num_proc=self.nproc)

		return output_dataset_truncated_w_length

if __name__ == "__main__":
	#tk = ST_TranscriptomeTokenizer({"study": "study"},nproc=32)
	#tk = ST_TranscriptomeTokenizer(nproc=32)
	tk = ST_TranscriptomeTokenizer({"study": "study", "Tissue": "Tissue", "Sample ID": "Sample ID", "Preparation": "Preparation", "barcode":"barcode", "coord_x":"coord_x", "coord_y":"coord_y"},nproc=32)
	#tk = ST_TranscriptomeTokenizer({"Tissue": "Tissue", "Sample ID": "Sample ID", "Preparation": "Preparation", "barcode":"barcode", "coord_x":"coord_x", "coord_y":"coord_y"},nproc=32)
	tk.tokenize_data("/home/qian/perturbmap/lung_sample", 
					 "/home/qian/perturbmap", 
					 "Cancer.Extended.Lung.dataset")
	
