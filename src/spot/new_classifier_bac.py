import datetime
import logging
import os
import pickle
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm, trange
from transformers import Trainer
from transformers.training_args import TrainingArguments
TOKEN_DICTIONARY_FILE = "/media/qian/vol2/Qian/models.to.test/Extended.model/SpatialModel/new_token_dictionary.pickle"
from collator_for_classification import (
DataCollatorForCellClassification,
DataCollatorForGeneClassification,
)
import classifier_utils as cu
import evaluation_utils as eu
import perturber_utils as pu
sns.set()
logger = logging.getLogger(__name__)
class Classifier:
	valid_option_dict = {
		"classifier": {"cell", "gene"},
		"cell_state_dict": {None, dict},
		"gene_class_dict": {None, dict},
		"filter_data": {None, dict},
		"rare_threshold": {int, float},
		"max_ncells": {None, int},
		"max_ncells_per_class": {None, int},
		"training_args": {None, dict},
		"freeze_layers": {int},
		"num_crossval_splits": {0, 1, 5},
		"split_sizes": {None, dict},
		"no_eval": {bool},
		"forward_batch_size": {int},
		"token_dictionary_file": {None, str},
		"nproc": {int},
		"ngpu": {int},
		"cust_id_class_dict": {None, dict},
	}
	def __init__(
		self,
		classifier=None,
		cell_state_dict=None,
		gene_class_dict=None,
		filter_data=None,
		rare_threshold=0,
		max_ncells=None,
		max_ncells_per_class=None,
		training_args=None,
		ray_config=None,
		freeze_layers=0,
		num_crossval_splits=1,
		split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
		no_eval=False,
		forward_batch_size=100,
		token_dictionary_file=None,
		nproc=4,
		ngpu=1,
		cust_id_class_dict=None,
	):
		self.classifier = classifier
		if self.classifier == "cell":
			self.model_type = "CellClassifier"
		elif self.classifier == "gene":
			self.model_type = "GeneClassifier"
		self.cell_state_dict = cell_state_dict
		self.gene_class_dict = gene_class_dict
		self.filter_data = filter_data
		self.rare_threshold = rare_threshold
		self.max_ncells = max_ncells
		self.max_ncells_per_class = max_ncells_per_class
		self.training_args = training_args
		self.ray_config = ray_config
		self.freeze_layers = freeze_layers
		self.num_crossval_splits = num_crossval_splits
		self.split_sizes = split_sizes
		self.train_size = self.split_sizes["train"]
		self.valid_size = self.split_sizes["valid"]
		self.oos_test_size = self.split_sizes["test"]
		self.eval_size = self.valid_size / (self.train_size + self.valid_size)
		self.no_eval = no_eval
		self.forward_batch_size = forward_batch_size
		self.token_dictionary_file = token_dictionary_file
		self.nproc = nproc
		self.ngpu = ngpu
		self.cust_id_class_dict = cust_id_class_dict
		if self.training_args is None:
			logger.warning(
				"Hyperparameter tuning is highly recommended for optimal results. "
				"No training_args provided; using default hyperparameters.")
		self.validate_options()
		if self.filter_data is None:
			self.filter_data = dict()
		if self.classifier == "cell":
			if self.cell_state_dict["states"] != "all":
				self.filter_data[self.cell_state_dict["state_key"]] = self.cell_state_dict["states"]
		# load token dictionary (Ensembl IDs:token)
		if self.token_dictionary_file is None:
			self.token_dictionary_file = TOKEN_DICTIONARY_FILE
		with open(self.token_dictionary_file, "rb") as f:
			self.gene_token_dict = pickle.load(f)

		self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}

		# filter genes for gene classification for those in token dictionary
		if self.classifier == "gene":
			all_gene_class_values = set(pu.flatten_list(self.gene_class_dict.values()))
			missing_genes = [gene for gene in all_gene_class_values if gene not in self.gene_token_dict.keys()]
			if len(missing_genes) == len(all_gene_class_values):
				logger.error("None of the provided genes to classify are in token dictionary.")
				raise
			elif len(missing_genes) > 0:
				logger.warning(f"Genes to classify {missing_genes} are not in token dictionary.")
			self.gene_class_dict = {
				k: list(set([self.gene_token_dict.get(gene) for gene in v]))
				for k, v in self.gene_class_dict.items()}
			empty_classes = []
			for k, v in self.gene_class_dict.items():
				if len(v) == 0:
					empty_classes += [k]
			if len(empty_classes) > 0:
				logger.error(f"Class(es) {empty_classes} did not contain any genes in the token dictionary.")
				raise

	def validate_options(self):
		# confirm arguments are within valid options and compatible with each other
		for attr_name, valid_options in self.valid_option_dict.items():
			attr_value = self.__dict__[attr_name]
			if not isinstance(attr_value, (list, dict)):
				if attr_value in valid_options:
					continue
			valid_type = False
			for option in valid_options:
				if (option in [int, float, list, dict, bool, str]) and isinstance(attr_value, option):
					valid_type = True
					break
			if valid_type:
				continue
			logger.error(
				f"Invalid option for {attr_name}. "
				f"Valid options for {attr_name}: {valid_options}")
			raise
		if self.filter_data is not None:
			for key, value in self.filter_data.items():
				if not isinstance(value, list):
					self.filter_data[key] = [value]
					logger.warning(
						"Values in filter_data dict must be lists. "
						f"Changing {key} value to list ([{value}]).")
		if self.classifier == "cell":
			if set(self.cell_state_dict.keys()) != set(["state_key", "states"]):
				logger.error(
					"Invalid keys for cell_state_dict. "
					"The cell_state_dict should have only 2 keys: state_key and states")
				raise
			if self.cell_state_dict["states"] != "all":
				if not isinstance(self.cell_state_dict["states"], list):
					logger.error("States in cell_state_dict should be list of states to model.")
					raise
				if len(self.cell_state_dict["states"]) < 2:
					logger.error("States in cell_state_dict should contain at least 2 states to classify.")
					raise
		if self.classifier == "gene":
			if len(self.gene_class_dict.keys()) < 2:
				logger.error("Gene_class_dict should contain at least 2 gene classes to classify.")
				raise
		if sum(self.split_sizes.values()) != 1:
			logger.error("Train, validation, and test proportions should sum to 1.")
			raise

	def prepare_data(self,
		input_data_file,
		#output_directory,
		o_dir,
		#output_prefix,
		o_prefix,
		split_id_dict=None,
		test_size=None,
		attr_to_split=None,
		attr_to_balance=None,
		max_trials=100,
		pval_threshold=0.1,
	):
		if test_size is None:
			test_size = self.oos_test_size
		# prepare data and labels for classification
		data = pu.load_and_filter(self.filter_data, self.nproc, input_data_file)
		if self.classifier == "cell":
			if "label" in data.features:
				logger.error("Column name 'label' must be reserved for class IDs. Please rename column.")
				raise
		elif self.classifier == "gene":
			if "labels" in data.features:
				logger.error("Column name 'labels' must be reserved for class IDs. Please rename column.")
				raise
		if (attr_to_split is not None) and (attr_to_balance is None):
			logger.error(
				"Splitting by attribute while balancing confounders requires both attr_to_split and attr_to_balance to be defined.")
			raise
		if not isinstance(attr_to_balance, list):
			attr_to_balance = [attr_to_balance]
		if self.classifier == "cell":
			# remove cell states representing < rare_threshold of cells
			data = cu.remove_rare(data, self.rare_threshold, self.cell_state_dict["state_key"], self.nproc)
			# downsample max cells and max per class
			data = cu.downsample_and_shuffle(data, self.max_ncells, self.max_ncells_per_class, self.cell_state_dict)
			# rename cell state column to "label"
			data = cu.rename_cols(data, self.cell_state_dict["state_key"])
			# convert classes to numerical labels and save as id_class_dict
			data, id_class_dict = cu.label_classes(
				self.classifier, data, self.cell_state_dict, self.nproc, self.cust_id_class_dict)
		elif self.classifier == "gene":
			data, id_class_dict = cu.label_classes(
				self.classifier, data, self.gene_class_dict, self.nproc, self.cust_id_class_dict)

		# save id_class_dict for future reference
		id_class_output_path = (Path(o_dir) / f"{o_prefix}_id_class_dict.pkl")
		with open(id_class_output_path, "wb") as f:
			pickle.dump(id_class_dict, f)
		if split_id_dict is not None:
			print("Entered here1")
			data_dict = dict()
			data_dict["train"] = pu.filter_by_dict(
				data, {split_id_dict["attr_key"]: split_id_dict["train"]}, self.nproc)
			data_dict["test"] = pu.filter_by_dict(
				data, {split_id_dict["attr_key"]: split_id_dict["test"]}, self.nproc)
			train_data_output_path = (Path(o_dir) / f"{o_prefix}_labeled_train.dataset")
			test_data_output_path = (Path(o_dir) / f"{o_prefix}_labeled_test.dataset")
			data_dict["train"].save_to_disk(str(train_data_output_path))
			data_dict["test"].save_to_disk(str(test_data_output_path))
		elif (test_size is not None) and (self.classifier == "cell"):
			print("Entered here2")
			if 1 > test_size > 0:
				if attr_to_split is None:
					data_dict = data.train_test_split(test_size=test_size, seed=42)
					train_data_output_path = (Path(o_dir) / f"{o_prefix}_labeled_train.dataset")
					test_data_output_path = (Path(o_dir) / f"{o_prefix}_labeled_test.dataset")
					data_dict["train"].save_to_disk(str(train_data_output_path))
					data_dict["test"].save_to_disk(str(test_data_output_path))
				else:
					data_dict, balance_df = cu.balance_attr_splits(data, attr_to_split,
						attr_to_balance, test_size, max_trials, pval_threshold,
						self.cell_state_dict["state_key"], self.nproc)
					balance_df.to_csv(f"{o_dir}/{o_prefix}_train_test_balance_df.csv")
					train_data_output_path = (Path(o_dir) / f"{o_prefix}_labeled_train.dataset")
					test_data_output_path = (Path(o_dir) / f"{o_prefix}_labeled_test.dataset")
					data_dict["train"].save_to_disk(str(train_data_output_path))
					data_dict["test"].save_to_disk(str(test_data_output_path))
			else:
				data_output_path = (Path(o_dir) / f"{o_prefix}_labeled.dataset")
				data.save_to_disk(str(data_output_path))
				print(data_output_path)
		else:
			print("Entered here3")
			data_output_path = (Path(o_dir) / f"{o_prefix}_labeled.dataset")
			data.save_to_disk(str(data_output_path))

	def train_all_data(
		self,
		model_directory,
		prepared_input_data_file,
		id_class_dict_file,
		#output_directory,
		o_dir,
		#output_prefix,
		o_prefix,
		save_eval_output=True,
		gene_balance=False,
	):
		if (gene_balance is True) and (len(self.gene_class_dict.values()) != 2):
			logger.error("Automatically balancing gene sets for training is only available for binary gene classifications.")
			raise
		##### Load data and prepare output directory #####
		# load numerical id to class dictionary (id:class)
		with open(id_class_dict_file, "rb") as f:
			id_class_dict = pickle.load(f)
		class_id_dict = {v: k for k, v in id_class_dict.items()}
		# load previously filtered and prepared data
		data = pu.load_and_filter(None, self.nproc, prepared_input_data_file)
		data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data
		# define output directory path
		current_date = datetime.datetime.now()
		datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
		if o_dir[-1:] != "/":  # add slash for dir if not present
			o_dir = o_dir + "/"
		output_dir = f"{o_dir}{datestamp}_cancerstformer_{self.classifier}Classifier_{o_prefix}/"
		subprocess.call(f"mkdir {output_dir}", shell=True)
		# get number of classes for classifier
		num_classes = cu.get_num_classes(id_class_dict)
		if self.classifier == "gene":
			targets = pu.flatten_list(self.gene_class_dict.values())
			labels = pu.flatten_list([[class_id_dict[label]] * len(targets)
				for label, targets in self.gene_class_dict.items()])
			assert len(targets) == len(labels)
			data = cu.prep_gene_classifier_all_data(data, targets, labels, 
				self.max_ncells, self.nproc, gene_balance)
		trainer = self.train_classifier(model_directory, num_classes, data, None, output_dir)
		return trainer

	def validate(
		self,
		model_directory,
		prepared_input_data_file,
		id_class_dict_file,
		#output_directory,
		o_dir,
		#output_prefix,
		o_prefix,
		split_id_dict=None,
		attr_to_split=None,
		attr_to_balance=None,
		gene_balance=False,
		max_trials=100,
		pval_threshold=0.1,
		save_eval_output=True,
		predict_eval=True,
		predict_trainer=False,
		n_hyperopt_trials=0,
		save_gene_split_datasets=True,
		debug_gene_split_datasets=False,
	):
		if self.num_crossval_splits == 0:
			logger.error("num_crossval_splits must be 1 or 5 to validate.")
			raise
		if (gene_balance is True) and (len(self.gene_class_dict.values()) != 2):
			logger.error("Automatically balancing gene sets for training is only available for binary gene classifications.")
			raise
		# ensure number of genes in each class is > 5 if validating model
		if self.classifier == "gene":
			insuff_classes = [k for k, v in self.gene_class_dict.items() if len(v) < 5]
			if (self.num_crossval_splits > 0) and (len(insuff_classes) > 0):
				logger.error(f"Insufficient # of members in class(es) {insuff_classes} to (cross-)validate.")
				raise
		##### Load data and prepare output directory #####
		# load numerical id to class dictionary (id:class)
		with open(id_class_dict_file, "rb") as f:
			id_class_dict = pickle.load(f)
		class_id_dict = {v: k for k, v in id_class_dict.items()}

		# load previously filtered and prepared data
		data = pu.load_and_filter(None, self.nproc, prepared_input_data_file)
		data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data

		# define output directory path
		current_date = datetime.datetime.now()
		datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
		if o_dir[-1:] != "/":  # add slash for dir if not present
			o_dir = o_dir + "/"
		output_dir = f"{o_dir}{datestamp}_cancerstformer_{self.classifier}Classifier_{o_prefix}/"
		subprocess.call(f"mkdir {output_dir}", shell=True)

		# get number of classes for classifier
		num_classes = cu.get_num_classes(id_class_dict)
		##### (Cross-)validate the model #####
		results = []
		all_conf_mat = np.zeros((num_classes, num_classes))
		iteration_num = 1
		if self.classifier == "cell":
			for i in trange(self.num_crossval_splits):
				print(f"****** Validation split: {iteration_num}/{self.num_crossval_splits} ******\n")
				ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
				if self.num_crossval_splits == 1:
					# single 1-eval_size:eval_size split
					if split_id_dict is not None:
						data_dict = dict()
						data_dict["train"] = pu.filter_by_dict(data, 
							{split_id_dict["attr_key"]: split_id_dict["train"]}, self.nproc)
						data_dict["test"] = pu.filter_by_dict(data,
							{split_id_dict["attr_key"]: split_id_dict["eval"]}, self.nproc)
					elif attr_to_split is not None:
						data_dict, balance_df = cu.balance_attr_splits(data, attr_to_split,
							attr_to_balance, self.eval_size, max_trials, pval_threshold,
							self.cell_state_dict["state_key"], self.nproc)
						balance_df.to_csv(f"{output_dir}/{o_prefix}_train_valid_balance_df.csv")
					else:
						data_dict = data.train_test_split(test_size=self.eval_size, seed=42)
					train_data = data_dict["train"]
					eval_data = data_dict["test"]
				else:
					# 5-fold cross-validate
					num_cells = len(data)
					fifth_cells = int(np.floor(num_cells * 0.2))
					num_eval = min((self.eval_size * num_cells), fifth_cells)
					start = i * fifth_cells
					end = start + num_eval
					eval_indices = [j for j in range(start, end)]
					train_indices = [j for j in range(num_cells) if j not in eval_indices]
					eval_data = data.select(eval_indices)
					train_data = data.select(train_indices)
				if n_hyperopt_trials == 0:
					trainer = self.train_classifier(model_directory, num_classes, train_data,
						eval_data, ksplit_output_dir, predict_trainer)
				else:
					trainer = self.hyperopt_classifier(model_directory, num_classes,
						train_data, eval_data, ksplit_output_dir, n_trials=n_hyperopt_trials)
					if iteration_num == self.num_crossval_splits:
						return
					else:
						iteration_num = iteration_num + 1
						continue

				result = self.evaluate_model(trainer.model, num_classes, id_class_dict, 
					eval_data, predict_eval, ksplit_output_dir, o_prefix)
				results += [result]
				all_conf_mat = all_conf_mat + result["conf_mat"]
				iteration_num = iteration_num + 1

		elif self.classifier == "gene":
			# set up (cross-)validation splits
			targets = pu.flatten_list(self.gene_class_dict.values())
			labels = pu.flatten_list([[class_id_dict[label]] * len(targets) 
				for label, targets in self.gene_class_dict.items()])
			assert len(targets) == len(labels)
			n_splits = int(1 / (1 - self.train_size))
			skf = cu.StratifiedKFold3(n_splits=n_splits, random_state=0, shuffle=True)
			# (Cross-)validate
			test_ratio = self.oos_test_size / (self.eval_size + self.oos_test_size)
			for train_index, eval_index, test_index in tqdm(
				skf.split(targets, labels, test_ratio)
			):
				print("Current", train_index, " ------ ", eval_index, " ------ ", test_index)
				print(f"****** Validation split: {iteration_num}/{self.num_crossval_splits} ******\n")
				ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")

				train_data, eval_data = cu.prep_gene_classifier_train_eval_split(
					data, targets, labels, train_index, eval_index, self.max_ncells,
					iteration_num, self.nproc, gene_balance)

				if save_gene_split_datasets is True:
					for split_name in ["train", "valid"]:
						labeled_dataset_output_path = (Path(output_dir)
							/ f"{o_prefix}_{split_name}_gene_labeled_ksplit{iteration_num}.dataset")
						if split_name == "train":
							train_data.save_to_disk(str(labeled_dataset_output_path))
						elif split_name == "valid":
							eval_data.save_to_disk(str(labeled_dataset_output_path))

				if self.oos_test_size > 0:
					test_data = cu.prep_gene_classifier_split(
						data, targets, labels, test_index, "test", self.max_ncells,
						iteration_num, self.nproc)
					if save_gene_split_datasets is True:
						test_labeled_dataset_output_path = (Path(output_dir)
							/ f"{o_prefix}_test_gene_labeled_ksplit{iteration_num}.dataset")
						test_data.save_to_disk(str(test_labeled_dataset_output_path))
				if debug_gene_split_datasets is True:
					logger.error(
						"Exiting after saving gene split datasets given debug_gene_split_datasets = True.")
					raise
				if n_hyperopt_trials == 0:
					trainer = self.train_classifier(model_directory, num_classes, train_data, 
						eval_data, ksplit_output_dir, predict_trainer)
					result = self.evaluate_model(trainer.model, num_classes, id_class_dict,
						eval_data, predict_eval, ksplit_output_dir, o_prefix)
				else:
					trainer = self.hyperopt_classifier(model_directory, num_classes, 
						train_data, eval_data, ksplit_output_dir, n_trials=n_hyperopt_trials)
					model = cu.load_best_model(ksplit_output_dir, self.model_type, 
						num_classes)
					if self.oos_test_size > 0:
						result = self.evaluate_model(model, num_classes, id_class_dict,
							test_data, predict_eval, ksplit_output_dir, o_prefix)
					else:
						if iteration_num == self.num_crossval_splits:
							return
						else:
							iteration_num = iteration_num + 1
							continue
				results += [result]
				all_conf_mat = all_conf_mat + result["conf_mat"]
				# break after 1 or 5 splits, each with train/eval proportions dictated by eval_size
				if iteration_num == self.num_crossval_splits:
					break
				iteration_num = iteration_num + 1

		all_conf_mat_df = pd.DataFrame(
			all_conf_mat, columns=id_class_dict.values(), index=id_class_dict.values())
		all_metrics = {"conf_matrix": all_conf_mat_df, 
			"macro_f1": [result["macro_f1"] for result in results],
			"acc": [result["acc"] for result in results]}
		all_roc_metrics = None  # roc metrics not reported for multiclass
		if num_classes == 2:
			mean_fpr = np.linspace(0, 1, 100)
			all_tpr = [result["roc_metrics"]["interp_tpr"] for result in results]
			all_roc_auc = [result["roc_metrics"]["auc"] for result in results]
			all_tpr_wt = [result["roc_metrics"]["tpr_wt"] for result in results]
			mean_tpr, roc_auc, roc_auc_sd = eu.get_cross_valid_roc_metrics(
				all_tpr, all_roc_auc, all_tpr_wt)
			all_roc_metrics = {"mean_tpr": mean_tpr, "mean_fpr": mean_fpr,
				"all_roc_auc": all_roc_auc, "roc_auc": roc_auc, "roc_auc_sd": roc_auc_sd}

		all_metrics["all_roc_metrics"] = all_roc_metrics
		if save_eval_output is True:
			eval_metrics_output_path = (Path(output_dir) / f"{o_prefix}_eval_metrics_dict.pkl")
			with open(eval_metrics_output_path, "wb") as f:
				pickle.dump(all_metrics, f)
		return all_metrics

	def hyperopt_classifier(
		self,
		model_directory,
		num_classes,
		train_data,
		eval_data,
		#output_directory,
		o_dir,
		n_trials=100,
	):
		# initiate runtime environment for raytune
		import ray
		from ray import tune
		from ray.tune.search.hyperopt import HyperOptSearch
		ray.shutdown()  # engage new ray session
		ray.init()
		##### Validate and prepare data #####
		train_data, eval_data = cu.validate_and_clean_cols(
			train_data, eval_data, self.classifier)
		if (self.no_eval is True) and (eval_data is not None):
			logger.warning("no_eval set to True; hyperparameter optimization requires eval, proceeding with eval")
		# ensure not overwriting previously saved model
		saved_model_test = os.path.join(o_dir, "pytorch_model.bin")
		if os.path.isfile(saved_model_test) is True:
			logger.error("Model already saved to this designated output directory.")
			raise
		# make output directory
		subprocess.call(f"mkdir {o_dir}", shell=True)
		##### Load model and training args #####
		model = pu.load_model(self.model_type, num_classes, model_directory, "train")

		def_training_args, def_freeze_layers = cu.get_default_train_args(
			model, self.classifier, train_data, o_dir)
		del model

		if self.training_args is not None:
			def_training_args.update(self.training_args)
		logging_steps = round(
			len(train_data) / def_training_args["per_device_train_batch_size"] / 10)
		def_training_args["logging_steps"] = logging_steps
		def_training_args["output_dir"] = o_dir
		if eval_data is None:
			def_training_args["evaluation_strategy"] = "no"
			def_training_args["load_best_model_at_end"] = False
		# only save last model for each run
		def_training_args.update({"save_strategy": "epoch", "save_total_limit": 1})  
		training_args_init = TrainingArguments(**def_training_args)

		##### Fine-tune the model #####
		# define the data collator
		if self.classifier == "cell":
			data_collator = DataCollatorForCellClassification(
				token_dictionary=self.gene_token_dict)
		elif self.classifier == "gene":
			data_collator = DataCollatorForGeneClassification(
				token_dictionary=self.gene_token_dict)
		# define function to initiate model
		def model_init():
			model = pu.load_model(self.model_type, num_classes, model_directory, "train")
			if self.freeze_layers is not None:
				def_freeze_layers = self.freeze_layers
			if def_freeze_layers > 0:
				modules_to_freeze = model.bert.encoder.layer[:def_freeze_layers]
				for module in modules_to_freeze:
					for param in module.parameters():
						param.requires_grad = False
			model = model.to("cuda:0")
			return model

		# create the trainer
		trainer = Trainer(model_init=model_init, args=training_args_init, data_collator=data_collator,
			train_dataset=train_data, eval_dataset=eval_data, compute_metrics=cu.compute_metrics)

		# specify raytune hyperparameter search space
		if self.ray_config is None:
			logger.warning(
				"No ray_config provided. Proceeding with default, but ranges may need adjustment depending on model.")
			def_ray_config = {
				"num_train_epochs": tune.choice([1]),
				"learning_rate": tune.loguniform(1e-6, 1e-3),
				"weight_decay": tune.uniform(0.0, 0.35),
				"lr_scheduler_type": tune.choice(["linear", "cosine", "polynomial"]),
				"warmup_steps": tune.uniform(100, 2000),
				"seed": tune.uniform(0, 100),
				"per_device_train_batch_size": tune.choice(
					[def_training_args["per_device_train_batch_size"]])}
		else:
			dx = self.ray_config
			def_ray_config = {
				"num_train_epochs": tune.choice(dx["num_train_epochs"]), 
				"learning_rate": tune.loguniform(dx["learning_rate"][0], dx["learning_rate"][1]),
				"weight_decay": tune.uniform(dx["weight_decay"][0], dx["weight_decay"][1]),
				"lr_scheduler_type": tune.choice(dx["lr_scheduler_type"]),
				"warmup_steps": tune.uniform(dx["warmup_steps"][0], dx["warmup_steps"][1]),
				"seed": tune.uniform(dx["seed"][0], dx["seed"][1]),
				"per_device_train_batch_size": tune.choice(dx["per_device_train_batch_size"])}
		#hyperopt_search = HyperOptSearch(metric="eval_roc_auc", mode="max")
		hyperopt_search = HyperOptSearch(metric="eval_macro_f1", mode="max")
		# optimize hyperparameters
		trainer.hyperparameter_search(
			direction="maximize",
			backend="ray",
			resources_per_trial={"cpu": int(self.nproc / self.ngpu), "gpu": 1},
			hp_space=lambda _: def_ray_config,
			search_alg=hyperopt_search,
			n_trials=n_trials,  # number of trials
			#metric="eval_roc_auc",
			#metric_columns=["loss", "eval_loss", "eval_accuracy", "eval_macro_f1", "eval_roc_auc"],
			progress_reporter=tune.CLIReporter(max_report_frequency=600, sort_by_metric=True,
				max_progress_rows=n_trials, mode="max", metric="eval_macro_f1", 
				metric_columns=["loss", "eval_loss", "eval_accuracy", "eval_macro_f1"]),
			storage_path=o_dir)
		return trainer

	def train_classifier(
		self,
		model_directory,
		num_classes,
		train_data,
		eval_data,
		#output_directory,
		o_dir,
		predict=False,
	):
		##### Validate and prepare data #####
		train_data, eval_data = cu.validate_and_clean_cols(
			train_data, eval_data, self.classifier)
		if (self.no_eval is True) and (eval_data is not None):
			logger.warning("no_eval set to True; model will be trained without evaluation.")
			eval_data = None
		if (self.classifier == "gene") and (predict is True):
			logger.warning(
				"Predictions during training not currently available for gene classifiers; setting predict to False.")
			predict = False
		# ensure not overwriting previously saved model
		saved_model_test = os.path.join(o_dir, "pytorch_model.bin")
		if os.path.isfile(saved_model_test) is True:
			logger.error("Model already saved to this designated output directory.")
			raise
		# make output directory
		subprocess.call(f"mkdir {o_dir}", shell=True)

		##### Load model and training args #####
		model = pu.load_model(self.model_type, num_classes, model_directory, "train")

		def_training_args, def_freeze_layers = cu.get_default_train_args(
			model, self.classifier, train_data, o_dir)

		if self.training_args is not None:
			def_training_args.update(self.training_args)
		logging_steps = round(
			len(train_data) / def_training_args["per_device_train_batch_size"] / 10)

		def_training_args["logging_steps"] = logging_steps
		def_training_args["output_dir"] = o_dir
		if eval_data is None:
			def_training_args["evaluation_strategy"] = "no"
			def_training_args["load_best_model_at_end"] = False
		training_args_init = TrainingArguments(**def_training_args)

		if self.freeze_layers is not None:
			def_freeze_layers = self.freeze_layers
		if def_freeze_layers > 0:
			modules_to_freeze = model.bert.encoder.layer[:def_freeze_layers]
			for module in modules_to_freeze:
				for param in module.parameters():
					param.requires_grad = False
		##### Fine-tune the model #####
		if self.classifier == "cell":
			data_collator = DataCollatorForCellClassification(
				token_dictionary=self.gene_token_dict)
		elif self.classifier == "gene":
			data_collator = DataCollatorForGeneClassification(
				token_dictionary=self.gene_token_dict)
		# create the trainer
		trainer = Trainer(model=model, args=training_args_init,
			data_collator=data_collator, train_dataset=train_data,
			eval_dataset=eval_data, compute_metrics=cu.compute_metrics)
		# train the classifier
		trainer.train()
		trainer.save_model(o_dir)
		if predict is True:
			# make eval predictions and save predictions and metrics
			predictions = trainer.predict(eval_data)
			prediction_output_path = f"{o_dir}/predictions.pkl"
			with open(prediction_output_path, "wb") as f:
				pickle.dump(predictions, f)
			trainer.save_metrics("eval", predictions.metrics)
		return trainer

	def evaluate_model(self, model, num_classes, id_class_dict, eval_data,
		predict=False, o_dir=None, o_prefix=None):
		##### Evaluate the model #####
		labels = id_class_dict.keys()
		y_pred, y_true, logits_list = eu.classifier_predict(
			model, self.classifier, eval_data, self.forward_batch_size)
		conf_mat, macro_f1, acc, roc_metrics = eu.get_metrics(
			y_pred, y_true, logits_list, num_classes, labels)
		if predict is True:
			pred_dict = {"pred_ids": y_pred, "label_ids": y_true, "predictions": logits_list}
			pred_dict_output_path = (Path(o_dir) / f"{o_prefix}_pred_dict.pkl")
			with open(pred_dict_output_path, "wb") as f:
				pickle.dump(pred_dict, f)
		return {"conf_mat": conf_mat, "macro_f1": macro_f1, "acc": acc, 
			"roc_metrics": roc_metrics}

	def evaluate_saved_model(self, model_directory, id_class_dict_file, test_data_file,
		o_dir, o_prefix, predict=True):
		with open(id_class_dict_file, "rb") as f:
			id_class_dict = pickle.load(f)
		num_classes = cu.get_num_classes(id_class_dict)
		test_data = pu.load_and_filter(None, self.nproc, test_data_file)
		model = pu.load_model(self.model_type, num_classes, model_directory, "eval")
		result = self.evaluate_model(model, num_classes, id_class_dict, test_data,
			predict=predict, o_dir=o_dir, o_prefix=o_prefix)
		all_conf_mat_df = pd.DataFrame(result["conf_mat"], columns=id_class_dict.values(),
			index=id_class_dict.values())
		all_metrics = {"conf_matrix": all_conf_mat_df, "macro_f1": result["macro_f1"],
			"acc": result["acc"]}
		all_roc_metrics = None  # roc metrics not reported for multiclass
		if num_classes == 2:
			mean_fpr = np.linspace(0, 1, 100)
			mean_tpr = result["roc_metrics"]["interp_tpr"]
			all_roc_auc = result["roc_metrics"]["auc"]
			all_roc_metrics = {"mean_tpr": mean_tpr, "mean_fpr": mean_fpr, 
				"all_roc_auc": all_roc_auc}
		all_metrics["all_roc_metrics"] = all_roc_metrics
		test_metrics_output_path = (Path(o_dir) / f"{o_prefix}_test_metrics_dict.pkl")
		with open(test_metrics_output_path, "wb") as f:
			pickle.dump(all_metrics, f)
		return all_metrics

	def plot_conf_mat(self, conf_mat_dict, o_dir, o_prefix,
		custom_class_order=None):
		for model_name in conf_mat_dict.keys():
			eu.plot_confusion_matrix(conf_mat_dict[model_name],
				model_name, o_dir, o_prefix, custom_class_order)

	def plot_roc(self, roc_metric_dict, model_style_dict, title, o_dir, o_prefix):
		eu.plot_ROC(roc_metric_dict, model_style_dict, title, o_dir, o_prefix)

	def plot_predictions(self, predictions_file, id_class_dict_file, title, o_dir,
		o_prefix, custom_class_order=None, kwargs_dict=None):
		with open(predictions_file, "rb") as f:
			predictions = pickle.load(f)
		with open(id_class_dict_file, "rb") as f:
			id_class_dict = pickle.load(f)
		if isinstance(predictions, dict):
			if all([key in predictions.keys() for key in ["pred_ids", "label_ids", "predictions"]]):
				# format is output from self.evaluate_saved_model
				predictions_logits = np.array(predictions["predictions"])
				true_ids = predictions["label_ids"]
		else:
			predictions_logits = predictions.predictions
			true_ids = predictions.label_ids
		num_classes = len(id_class_dict.keys())
		num_predict_classes = predictions_logits.shape[1]
		assert num_classes == num_predict_classes
		classes = id_class_dict.values()
		true_labels = [id_class_dict[idx] for idx in true_ids]
		predictions_df = pd.DataFrame(predictions_logits, columns=classes)
		if custom_class_order is not None:
			predictions_df = predictions_df.reindex(columns=custom_class_order)
		predictions_df["true"] = true_labels
		custom_dict = dict(zip(classes, [i for i in range(len(classes))]))
		if custom_class_order is not None:
			custom_dict = dict(
				zip(custom_class_order, [i for i in range(len(custom_class_order))]))
		predictions_df = predictions_df.sort_values(
			by=["true"], key=lambda x: x.map(custom_dict))
		eu.plot_predictions(
			predictions_df, title, o_dir, o_prefix, kwargs_dict)
