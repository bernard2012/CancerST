import torch
import numpy as np
import pickle
import os
import sys
from geneformer_emb_extractor import EmbExtractor
from geneformer_in_silico_perturber import InSilicoPerturber
from geneformer_in_silico_perturber_stats import InSilicoPerturberStats

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()

def run_perturb(model_path,dataset_path,out_dir,num_embs=None):
	#Import list of genes to perturb
	#genes_perturb = list(np.loadtxt("immune.gene.set", dtype=str))
	genes_perturb = list(np.loadtxt("lr.ligands.genes", dtype=str))
	#genes_perturb = list(np.loadtxt("CD40.query", dtype=str))
	
	file_path = "jan21_qian_gene_name_id_dictionary.pickle"

	# Open the file in read-binary mode and load it with pickle
	with open(file_path, 'rb') as file:
		ensemble_dictionary = pickle.load(file)
		
	# Translate gene names to Ensembl IDs using the token_dictionary
	good_genes = [ensemble_dictionary[gene] for gene in genes_perturb if gene in ensemble_dictionary]


	print("Good genes", len(good_genes), "out of", len(ensemble_dictionary.keys()))   
 
	file_path = "jan21_qian_new_token_dictionary.pickle"

	# Open the file in read-binary mode and load it with pickle
	with open(file_path, 'rb') as file:
		token_dictionary = pickle.load(file)
		
	# Translate gene names to Ensembl IDs using the token_dictionary
	good_genes_final = [gene for gene in good_genes if gene in list(token_dictionary.keys())]
	
	#Filter healthy only
	filter_data_dict={"Disease":["TNBC"]}

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
								model_type="Pretrained",
								num_classes=0,
								emb_mode="cell_and_gene",
								cell_emb_style="mean_pool",
								filter_data=filter_data_dict,
								max_ncells=1000,
								emb_layer=-1,
								forward_batch_size=50,
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
			
#out_dir = "/project/Geneformer/STGeneformer/Pertubation_Study/Results_Run1_GeneShift_indv_HEALTHY"
out_dir = "Checkpoint_Gene_Shift_Single_Gene_1000_lr_test"
os.makedirs(out_dir,exist_ok=True)
model_path = "models/"
#dataset_path = "/project/Geneformer/STGeneformer/Pertubation_Study/Dataset/STGeneformer_TNBC_Normal_Perturbset.dataset"
dataset_path = "STGeneformer_TNBC_Normal_Perturbset_filtered.dataset"

run_perturb(model_path,dataset_path,out_dir)


