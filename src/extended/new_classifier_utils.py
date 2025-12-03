import json
import logging
import os
import random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ranksums
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
import new_perturber_utils as pu
logger = logging.getLogger(__name__)

def downsample_and_shuffle(data, max_num_spots, max_num_spots_per_class, spot_state_dict):
	data = data.shuffle(seed=42)
	num_spots = len(data)
	# if max number of spots is defined, then subsample to this max number
	if max_num_spots is not None:
		if num_spots > max_num_spots:
			data = data.select([i for i in range(max_num_spots)])
	if max_num_spots_per_class is not None:
		class_labels = data[spot_state_dict["state_key"]]
		random.seed(42)
		subsample_indices = subsample_by_class(class_labels, max_num_spots_per_class)
		data = data.select(subsample_indices)
	return data

def subsample_by_class(labels, N):
	label_indices = defaultdict(list)
	for idx, label in enumerate(labels):
		label_indices[label].append(idx)
	selected_indices = []
	for label, indices in label_indices.items():
		if len(indices) > N:
			selected_indices.extend(random.sample(indices, N))
		else:
			selected_indices.extend(indices)
	return selected_indices

def rename_cols(data, state_key):
	data = data.rename_column(state_key, "label")
	return data

def validate_and_clean_cols(train_data, eval_data, classifier):
	if classifier == "spot":
		label_col = "label"
	elif classifier == "gene":
		label_col = "labels"
	cols_to_keep = [label_col] + ["input_ids", "length"]
	if label_col not in train_data.column_names:
		logger.error(f"train_data must contain column {label_col} with class labels.")
		raise
	else:
		train_data = remove_cols(train_data, cols_to_keep)
	if eval_data is not None:
		if label_col not in eval_data.column_names:
			logger.error(f"eval_data must contain column {label_col} with class labels.")
			raise
		else:
			eval_data = remove_cols(eval_data, cols_to_keep)
	return train_data, eval_data

def remove_cols(data, cols_to_keep):
	other_cols = list(data.features.keys())
	other_cols = [ele for ele in other_cols if ele not in cols_to_keep]
	data = data.remove_columns(other_cols)
	return data

def remove_rare(data, rare_threshold, label, nproc):
	if rare_threshold > 0:
		total_spots = len(data)
		label_counter = Counter(data[label])
		nonrare_label_dict = {
			label: [k for k, v in label_counter if (v / total_spots) > rare_threshold]
		}
		data = pu.filter_by_dict(data, nonrare_label_dict, nproc)
	return data

def label_classes(classifier, data, gene_class_dict, nproc, id_class_dict):
	if classifier == "spot":
		label_set = set(data["label"])
	elif classifier == "gene":
		# remove spots without any of the target genes
		def if_contains_label(example):
			a = pu.flatten(gene_class_dict.values())
			b = example["input_ids"]
			return not set(a).isdisjoint(b)
		data = data.filter(if_contains_label, num_proc=nproc)
		label_set = gene_class_dict.keys()
		if len(data) == 0:
			logger.error("No spots remain after filtering for target genes. Check target gene list.")
			raise
	print("id_class_dict", id_class_dict)
	print("gene_class_dict", gene_class_dict)
	sorted_ids = sorted(list(id_class_dict.keys()))
	label_set_good = [id_class_dict[k] for k in sorted_ids]
	for l in label_set:
		if l not in set(label_set_good):
			print("ERROR label mismatch", l)
			raise
	label_set = label_set_good
	print("label_set", label_set)
	class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
	#id_class_dict = {v: k for k, v in class_id_dict.items()}
	print("class_id_dict", class_id_dict)
	if classifier == "gene":
		inverse_gene_class_dict = {}
		for key, value_list in gene_class_dict.items():
			for value in value_list:
				inverse_gene_class_dict[value] = key
		print("inverse_gene_class_dict", inverse_gene_class_dict)
	def classes_to_ids(example):
		if classifier == "spot":
			example["label"] = class_id_dict[example["label"]]
		elif classifier == "gene":
			example["labels"] = label_gene_classes(example, class_id_dict, inverse_gene_class_dict)
		return example
	data = data.map(classes_to_ids, num_proc=nproc)
	return data, id_class_dict

def label_gene_classes(example, class_id_dict, inverse_gene_class_dict):
	first_len = 2048
	res = []
	for ind, token_id in enumerate(example["input_ids"]):
		if ind <= first_len: #central_node
			res.append(class_id_dict.get(inverse_gene_class_dict.get(token_id, -100), -100))
		else:
			res.append(-100)
	return res
	'''
	return [class_id_dict.get(inverse_gene_class_dict.get(token_id, -100), -100) for token_id in example["input_ids"]]
	'''
def prep_gene_classifier_train_eval_split(data, targets, labels, train_index, eval_index, max_num_spots,
	iteration_num, num_proc):
	# generate cross-validation splits
	train_data = prep_gene_classifier_split(data, targets, labels, train_index, "train",
		max_num_spots, iteration_num, num_proc)
	eval_data = prep_gene_classifier_split(data, targets, labels, eval_index, "eval",
		max_num_spots, iteration_num, num_proc)
	return train_data, eval_data

def prep_gene_classifier_split(data, targets, labels, index, subset_name, max_num_spots, iteration_num, num_proc):
	# generate cross-validation splits
	targets = np.array(targets)
	labels = np.array(labels)
	targets_subset = targets[index]
	labels_subset = labels[index]
	label_dict_subset = dict(zip(targets_subset, labels_subset))
	# function to filter by whether contains train or eval labels
	def if_contains_subset_label(example):
		a = targets_subset
		b = example["input_ids"]
		return not set(a).isdisjoint(b)
	# filter dataset for examples containing classes for this split
	logger.info(f"Filtering data for {subset_name} genes in split {iteration_num}")
	subset_data = data.filter(if_contains_subset_label, num_proc=num_proc)
	logger.info(f"Filtered {round((1-len(subset_data)/len(data))*100)}%; {len(subset_data)} remain\n")
	# subsample to max_num_spots
	subset_data = downsample_and_shuffle(subset_data, max_num_spots, None, None)
	# relabel genes for this split
	def subset_classes_to_ids(example):
		'''
		example["labels"] = [label_dict_subset.get(token_id, -100) for token_id in example["input_ids"]]
		'''
		first_len = 2048
		res = []
		for ind, token_id in enumerate(example["input_ids"]):
			if ind<=first_len:
				res.append(label_dict_subset.get(token_id, -100))
			else:
				res.append(-100)
		example["labels"] = res
		return example
	subset_data = subset_data.map(subset_classes_to_ids, num_proc=num_proc)
	return subset_data

def prep_gene_classifier_all_data(data, targets, labels, max_num_spots, num_proc):
	targets = np.array(targets)
	labels = np.array(labels)
	label_dict_train = dict(zip(targets, labels))
	# function to filter by whether contains train labels
	def if_contains_train_label(example):
		a = targets
		b = example["input_ids"]
		return not set(a).isdisjoint(b)
	# filter dataset for examples containing classes for this split
	logger.info("Filtering training data for genes to classify.")
	train_data = data.filter(if_contains_train_label, num_proc=num_proc)
	logger.info(f"Filtered {round((1-len(train_data)/len(data))*100)}%; {len(train_data)} remain\n")
	# subsample to max_num_spots
	train_data = downsample_and_shuffle(train_data, max_num_spots, None, None)
	# relabel genes for this split
	def train_classes_to_ids(example):
		'''
		example["labels"] = [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]]
		'''
		first_len = 2048
		res = []
		for ind,token_id in enumerate(example["input_ids"]):
			if ind<=first_len:
				res.append(label_dict_train.get(token_id, -100))
			else:
				res.append(-100)
		example["labels"] = res
		return example
	train_data = train_data.map(train_classes_to_ids, num_proc=num_proc)
	return train_data

def count_genes_for_balancing(subset_data, label_dict_subset, num_proc):
	def count_targets(example):
		'''
		labels = [label_dict_subset.get(token_id, -100) for token_id in example["input_ids"]]
		'''
		labels = []
		first_len = 2048
		for ind,token_id in enumerate(example["input_ids"]):
			if ind<=first_len:
				labels.append(label_dict_subset.get(token_id, -100))
			else:
				labels.append(-100)
		counter_labels = Counter(labels)
		example["labels_counts"] = [counter_labels.get(0, 0), counter_labels.get(1, 0)]
		return example
	subset_data = subset_data.map(count_targets, num_proc=num_proc)
	label0_counts = sum([counts[0] for counts in subset_data["labels_counts"]])
	label1_counts = sum([counts[1] for counts in subset_data["labels_counts"]])
	subset_data = subset_data.remove_columns("labels_counts")
	return label0_counts, label1_counts

def filter_data_balanced_genes(subset_data, label_dict_subset, num_proc):
	# function to filter by whether contains labels
	def if_contains_subset_label(example):
		a = list(label_dict_subset.keys())
		b = example["input_ids"]
		return not set(a).isdisjoint(b)
	# filter dataset for examples containing classes for this split
	logger.info("Filtering data for balanced genes")
	subset_data_len_orig = len(subset_data)
	subset_data = subset_data.filter(if_contains_subset_label, num_proc=num_proc)
	logger.info(f"Filtered {round((1-len(subset_data)/subset_data_len_orig)*100)}%; {len(subset_data)} remain\n")
	return subset_data, label_dict_subset

def get_num_classes(id_class_dict):
	return len(set(id_class_dict.values()))

def py_softmax(vector):
	e = np.exp(vector)
	return e / e.sum()

def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	# calculate accuracy and macro f1 using sklearn's function
	if len(labels.shape) == 1:
		acc = accuracy_score(labels, preds)
		macro_f1 = f1_score(labels, preds, average="macro")
	else:
		flat_labels = labels.flatten().tolist()
		flat_preds = preds.flatten().tolist()
		logit_label_paired = [item for item in list(zip(flat_preds, flat_labels)) if item[1] != -100]
		y_pred = [item[0] for item in logit_label_paired]
		y_true = [item[1] for item in logit_label_paired]
		#print("y_pred 1 length", len(y_pred), y_pred)
		#print("y_true 1 length", len(y_true), y_true)
		acc = accuracy_score(y_true, y_pred)
		macro_f1 = f1_score(y_true, y_pred, average="macro")
	#return {"accuracy": acc, "macro_f1": macro_f1, "roc_auc": roc_auc}
	return {"accuracy": acc, "macro_f1": macro_f1}

def get_default_train_args(model, classifier, data, output_dir):
	num_layers = pu.quant_layers(model)
	freeze_layers = 0
	batch_size = 12
	if classifier == "spot":
		epochs = 10
		evaluation_strategy = "epoch"
		load_best_model_at_end = True
	else:
		epochs = 1
		evaluation_strategy = "no"
		load_best_model_at_end = False
	if num_layers == 6:
		default_training_args = {"learning_rate": 5e-5, "lr_scheduler_type": "linear", "warmup_steps": 500,
			"per_device_train_batch_size": batch_size, "per_device_eval_batch_size": batch_size}
	else:
		default_training_args = {"per_device_train_batch_size": batch_size, "per_device_eval_batch_size": batch_size}
	training_args = {"num_train_epochs": epochs, "do_train": True, "do_eval": True,
		"evaluation_strategy": evaluation_strategy, "logging_steps": np.floor(len(data) / batch_size / 8),  # 8 evals per epoch
		"save_strategy": "epoch", "group_by_length": False, "length_column_name": "length", 
		"disable_tqdm": False, "weight_decay": 0.001, "load_best_model_at_end": load_best_model_at_end}
	training_args.update(default_training_args)
	return training_args, freeze_layers

def load_best_model(directory, model_type, num_classes, mode="eval"):
	file_dict = dict()
	#best_model_key = "eval_roc_auc" #or eval_macro_f1
	best_model_key = "eval_macro_f1" #or eval_macro_f1
	for subdir, dirs, files in os.walk(directory):
		for file in files:
			#if file.endswith("result.json"):
			if file.endswith("result.json") and os.stat(f"{subdir}/{file}").st_size>0:
				with open(f"{subdir}/{file}", "rb") as fp:
					result_json = json.load(fp)
				file_dict[f"{subdir}"] = result_json[best_model_key]
	file_df = pd.DataFrame({"dir": file_dict.keys(), best_model_key: file_dict.values()})
	model_superdir = ("run-" + file_df.iloc[file_df[best_model_key].idxmax()]["dir"].split("_objective_")[2].split("_")[0])
	for subdir, dirs, files in os.walk(f"{directory}/{model_superdir}"):
		for file in files:
			if file.endswith("model.safetensors"):
				model = pu.load_model(model_type, num_classes, f"{subdir}", mode)
	return model
class StratifiedKFold3(StratifiedKFold):
	def split(self, targets, labels, test_ratio=0.5, groups=None):
		s = super().split(targets, labels, groups)
		for train_indxs, test_indxs in s:
			if test_ratio == 0:
				yield train_indxs, test_indxs, None
			else:
				labels_test = np.array(labels)[test_indxs]
				valid_indxs, test_indxs = train_test_split(test_indxs, stratify=labels_test,
					test_size=test_ratio, random_state=0)
				yield train_indxs, valid_indxs, test_indxs
