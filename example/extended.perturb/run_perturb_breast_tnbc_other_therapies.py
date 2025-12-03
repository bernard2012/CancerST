import torch
import numpy as np
import pickle
import os
import sys
from new_emb_extractor import EmbExtractor
from new_in_silico_perturber import InSilicoPerturber
from new_in_silico_perturber_stats import InSilicoPerturberStats

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()
cur = "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/Extended.model"

def run_perturb(model_path,dataset_path,out_dir,num_embs=None):
	#Import list of genes to perturb
	genes_perturb = list(np.loadtxt("immune.gene.set", dtype=str, ndmin=1))
	#genes_perturb = list(np.loadtxt("tnbc.ligands", dtype=str)) #good
	#genes_perturb = list(np.loadtxt("checkpoint.combo", dtype=str)) #good
	#genes_perturb = list(np.loadtxt("CD40.query", dtype=str))

	file_path = "%s/SpatialModel/new_token_dictionary.pickle" % cur

	# Open the file in read-binary mode and load it with pickle
	with open(file_path, 'rb') as file:
		token_dictionary = pickle.load(file)
	
	#good_genes = [ensemble_dictionary[gene] for gene in genes_perturb if gene in ensemble_dictionary]
	# Translate gene names to Ensembl IDs using the token_dictionary
	good_genes_final = [gene for gene in genes_perturb if gene in list(token_dictionary.keys())]
	

	for i, gene in enumerate(good_genes_final):
		try:
			print(f"Perturbing {gene}")
			out_dir_final = out_dir + f"/{gene}"
			os.makedirs(out_dir_final,exist_ok=True)
			isp = InSilicoPerturber(perturb_type="delete",
								perturb_rank_shift=None,
								genes_to_perturb = [gene],
								combos=0,
								anchor_gene=None,
								#model_type="Pretrained",
								model_type="GeneClassifier",
								#num_classes=0,
								num_classes=2,
								emb_mode="spot_and_gene",
								spot_emb_style="mean_pool",
								#filter_data=filter_data_dict,
								max_num_spots=1000,
								emb_layer=-1,
								forward_batch_size=80,
								nproc=1)
			
			isp.perturb_data(model_path,
							 dataset_path,
							 out_dir_final,
							 os.path.basename(dataset_path).replace(".dataset","_emb"))
			
			ispstats = InSilicoPerturberStats(mode="aggregate_gene_shifts",
										  genes_perturbed = [gene],
										  combos=0,
										  anchor_gene=None)
			
			ispstats.get_stats(out_dir_final,
							   None,
							   out_dir_final,
							   os.path.basename(dataset_path).replace(".dataset","_emb"))
		except Exception as e:
			print(f"Error perturbing gene {gene}: {e}")
			
run_id = "run-8eb93bdf"
out_dir = "TNBC_Gene_Shift_Single_Gene_1000/%s" % run_id
os.makedirs(out_dir,exist_ok=True)
model_path = "%s/checkpoint-1000" % run_id
dataset_path = "STFormer_TNBC_neighbor.dataset"

run_perturb(model_path,dataset_path,out_dir)
