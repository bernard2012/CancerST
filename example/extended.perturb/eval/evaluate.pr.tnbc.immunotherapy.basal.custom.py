import sys
import os
import re
from scipy.stats import hypergeom
import numpy as np

def read_lr_targets(n):
	f = open(n)
	h = f.readline().rstrip("\n").split("\t")[1:]
	by_ligand = {}
	for l in f:
		l = l.rstrip("\n")
		ll = l.split("\t")
		lx = zip(h, ll[1:])
		for i1, i2 in lx:
			by_ligand.setdefault(i1, [])
			by_ligand[i1].append(i2)
	f.close()
	return by_ligand

def read_target_list(n):
	f = open(n)
	tt = []
	for l in f:
		l = l.rstrip("\n")
		tt.append(l.split("\t")[1])
	f.close()
	return tt

def pr_from_ranklist(ranklist, gold):
	tp = 0
	precisions, recalls = [], []
	G = len(gold)                       # number of true positives in gold standard
	for i, gene in enumerate(ranklist, 1):   # 1‑based rank
		if gene in gold:
			tp += 1
		precisions.append(tp / i)
		recalls.append(tp / G)
	return np.asarray(recalls), np.asarray(precisions)

def read_pert(n, checkpoint, detect_min=-1):
#,Perturbed,Gene_name,Ensembl_ID,Affected,Affected_gene_name,Affected_Ensembl_ID,Cosine_sim_mean,Cosine_sim_stdev,N_Detections
#0,4242,CD274,ENSG00000120217,cell_emb,,,0.9908808100819587,0.00434944449786566,1000
#4965,4242,CD274,ENSG00000120217,5588,GIMAP6,ENSG00000133561,0.7777600967708755,0.15686462345751515,136
#7245,4242,CD274,ENSG00000120217,12570,PCNX3,ENSG00000197136,0.8419126406495954,0.1376923238593469,162
#7244,4242,CD274,ENSG00000120217,12596,SYNGAP1,ENSG00000197283,0.8504492286089305,0.13708373694658088,3
	f = open(checkpoint + "/" + n + "/STFormer_TNBC_neighbor_emb.csv")
	h = f.readline().rstrip("\n").split(",")
	gene_list = []
	filtered = []
	for l in f:
		l = l.rstrip("\n")
		ll = l.split(",")
		ndetect = int(ll[-1])
		if detect_min!=-1:
			if ndetect<detect_min:
				filtered.append(ll[5])
				continue
		gene = ll[5]
		gene_list.append(gene)
	f.close()
	gene_list = gene_list + filtered
	return gene_list

def read_conversion(n):
	f = open(n)
	m = []
	for l in f:
		l = l.rstrip("\n")
		ll = l.split("\t")
		m.append((ll[0], ll[1]))
	f.close()
	return m

if __name__=="__main__":
	by_ligand = read_lr_targets("/media/scandisk/Perturbation/pdcd1.ispy2.basal.targets.txt")
	#print(by_ligand)
	target_list = read_target_list("../profiles.targets.txt")
	#pert_genes = read_pert("ENSG00000120217")

	checkpoint = sys.argv[1]

	#print("Target list", len(target_list))
	#print("Pert list", len(pert_genes))
	#print("Overlap list", len(set(target_list) & set(pert_genes)))
	fw1 = open("%s/TNBC_immunotherapy_basal_fold_pr_over_random.txt" % checkpoint, "w")
	fw2 = open("%s/TNBC_immunotherapy_basal_pr.txt" % checkpoint, "w")
	fw3 = open("%s/TNBC_immunotherapy_basal_recall.txt" % checkpoint, "w")

	#sym_ensembl = read_conversion("conversion")

	for sym in ["CD274", "PDCD1", "CTLA4"]:
		for direction in ["up", "down"]:
			pert_genes = read_pert(sym, checkpoint, detect_min=int(sys.argv[2]))
			print("Pert gene", sym)
			eval_genes = list(set(target_list) & set(pert_genes))

			l_target = [e for e in by_ligand["PDCD1_%s" % (direction)] if e in eval_genes]

			#predicted ligand targets
			pred = [e for e in pert_genes[:500] if e in eval_genes]
			ov = set(l_target) & set(pred)

			#print(len(ov), len(l_target), len(pred), len(eval_genes))
			N = len(eval_genes)
			K = len(l_target)
			n = len(pred)
			k = len(ov)

			p = hypergeom.sf(k-1, N, K, n)
			logp = -1.0 * np.log10(p)
	
			exp_k = hypergeom.isf(0.50, N, K, n) + 1

			fold_over_random = len(ov) / exp_k

			recall, precision = pr_from_ranklist(pert_genes, set(l_target))
			grid = np.arange(0.00, 1.01, 0.01)             # 0.00, 0.01, …, 1.00
			interp_prec = [precision[recall >= r].max() if np.any(recall >= r) else 0.0 for r in grid]
	
			baseline_pr = len(l_target) / len(pert_genes)
			fold_pr = [ip/baseline_pr for ip in interp_prec]

			fw1.write(" ".join(["%f" % fr for fr in fold_pr]) + "\n")
			fw2.write(" ".join(["%f" % pr for pr in interp_prec]) + "\n")
			fw3.write(" ".join(["%f" % re for re in grid]) + "\n")
		#print(len(ov), len(l_target), len(pred), len(eval_genes), logp, exp_k, fold_over_random)
	fw1.close()
	fw2.close()
	fw3.close()
