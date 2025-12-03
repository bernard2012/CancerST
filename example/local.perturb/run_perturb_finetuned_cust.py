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

def run_perturb(model_path,dataset_path,out_dir,num_embs=None):
    #Import list of genes to perturb
    genes_perturb = list(np.loadtxt("immune.gene.set", dtype=str))
    #genes_perturb = list(np.loadtxt("tnbc.ligands", dtype=str)) #good
    #genes_perturb = list(np.loadtxt("checkpoint.combo", dtype=str)) #good
    #genes_perturb = list(np.loadtxt("CD40.query", dtype=str))
    
    file_path = "jan21_qian_gene_name_id_dictionary.pickle"

    # Open the file in read-binary mode and load it with pickle
    with open(file_path, 'rb') as file:
        ensemble_dictionary = pickle.load(file)
        
    # Translate gene names to Ensembl IDs using the token_dictionary
    good_genes = [ensemble_dictionary[gene] for gene in genes_perturb if gene in ensemble_dictionary]
    
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
                                #model_type="Pretrained",
                                model_type="GeneClassifier",
                                #num_classes=0,
                                num_classes=2,
                                emb_mode="spot_and_gene",
                                spot_emb_style="mean_pool",
                                filter_data=filter_data_dict,
                                max_num_spots=1000,
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
#out_dir = "Checkpoint_Gene_Shift_Single_Gene_1000_maxncell_10000"

#datestamp = sys.argv[1]
#run_id = sys.argv[2]
run_id = "run-c3099a67"

out_dir = "TNBC_Gene_Shift_%s" % (run_id)
os.makedirs(out_dir,exist_ok=True)
#model_path = "models/"

'''
res_dir = None #250919_geneformer_geneClassifier_responder_test
for root, dirs, files in os.walk("finetuned"):
	for d in dirs:
		if d.endswith("responder_test"):
			res_dir = d
			break
'''
#251118071123/251118_cancerstformer_geneClassifier_responder_test/ksplit1/run-c3099a67/checkpoint-1000
cpt_dir = None
for root, dirs, files in os.walk("251118071123/251118_cancerstformer_geneClassifier_responder_test/ksplit1/%s" % (run_id)):
	for d in dirs:
		if d.startswith("checkpoint-"):
			cpt_dir = d
			break

model_path = "251118071123/251118_cancerstformer_geneClassifier_responder_test/ksplit1/%s/%s" % (run_id, cpt_dir)
dataset_path = "STGeneformer_TNBC_Normal_Perturbset_filtered.dataset"
#dataset_path = "STGeneformer.dataset"

run_perturb(model_path,dataset_path,out_dir)


