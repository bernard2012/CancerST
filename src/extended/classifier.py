import datetime
import logging
import os
import pickle
import subprocess
from itertools import compress
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

import classifier_utils as cu
import evaluation_utils as eu
import perturber_utils as pu
from collator_for_classification import DataCollatorForGeneClassification

logger = logging.getLogger(__name__)


class Classifier:
	def __init__(
		self,
		classifier="cell",
		cell_state_dict=None,
		gene_class_dict=None,
		feature_dict=None,
		filter_data=None,
		test_set_size=0.0,
		val_set_size=0.0,
		model_type="geneformer",
		training_args=None,
		quantize=False,
		max_cells_per_class=553500,
		num_classes=2,
		num_crossval_splits=1,
		split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
		stratify_splits_col=None,
		no_eval=False,
		forward_batch_size=100,
		token_dictionary_file=None,
		nproc=4,
		ngpu=1,
		cust_id_class_dict=None,
	):
		self.classifier = classifier
		self.model_type = model_type
		self.training_args = training_args
		self.quantize = quantize
		self.max_cells_per_class = max_cells_per_class
		self.num_classes = num_classes
		self.num_crossval_splits = num_crossval_splits
		self.split_sizes = split_sizes
		self.stratify_splits_col = stratify_splits_col
		self.no_eval = no_eval
		self.forward_batch_size = forward_batch_size
		self.token_dictionary_file = token_dictionary_file
		self.nproc = nproc
		self.ngpu = ngpu

		self.val_set_size = val_set_size
		self.test_set_size = test_set_size

		self.cell_state_dict = cell_state_dict
		self.gene_class_dict = gene_class_dict
		self.feature_dict = feature_dict
		self.filter_data = filter_data

		self.cust_id_class_dict = cust_id_class_dict

	def validate_options(self):
		if self.classifier not in ["cell", "gene"]:
			logger.error("Classifier must be 'cell' or 'gene'.")
			raise

		if (self.classifier == "gene") and (self.gene_class_dict is None):
			logger.error(
				"gene_class_dict must be provided for GeneClassifiers. See documentation for format."
			)
			raise

		if self.num_crossval_splits < 1:
			logger.error("num_crossval_splits must be >=1.")
			raise

	def prepare_data(
		self,
		tokenized_data_file,
		prepared_input_data_file,
		output_directory=None,
		output_prefix=None,
	):
		self.validate_options()

		data = pu.load_and_filter(
			self.filter_data,
			self.nproc,
			tokenized_data_file,
		)

		if self.classifier == "cell":
			if self.cell_state_dict is not None:
				data = cu.prepare_cell_states(self.cell_state_dict, data)

			if self.feature_dict is not None:
				data = cu.prepare_features(self.feature_dict, data)

		elif self.classifier == "gene":
			data = cu.prepare_gene_states(self.gene_class_dict, data)

		# save filtered and prepared data
		data.save_to_disk(prepared_input_data_file)

		# save labeled data splits
		if self.test_set_size > 0:
			if self.val_set_size > 0:
				test_size = self.test_set_size
				val_size = self.val_set_size / (1 - self.test_set_size)
			else:
				test_size = self.test_set_size
				val_size = 0
			data_dict = data.train_test_split(
				test_size=test_size,
				stratify_by_column=self.stratify_splits_col,
				seed=42,
			)
			if val_size > 0:
				test_data = data_dict["test"]
				test_val_dict = test_data.train_test_split(
					test_size=val_size,
					stratify_by_column=self.stratify_splits_col,
					seed=42,
				)
				data_dict["test"] = test_val_dict["test"]
				data_dict["valid"] = test_val_dict["train"]

			if output_directory is not None and output_prefix is not None:
				if self.test_set_size > 0:
					train_data_output_path = (
						Path(output_directory) / f"{output_prefix}_labeled_train"
					).with_suffix(".dataset")
					test_data_output_path = (
						Path(output_directory) / f"{output_prefix}_labeled_test"
					).with_suffix(".dataset")
					data_dict["train"].save_to_disk(str(train_data_output_path))
					data_dict["test"].save_to_disk(str(test_data_output_path))
			else:
				data_output_path = (
					Path(output_directory) / f"{output_prefix}_labeled"
				).with_suffix(".dataset")
				data.save_to_disk(str(data_output_path))
		else:
			data_output_path = prepared_input_data_file
			data.save_to_disk(str(data_output_path))

	def validate(
		self,
		model_directory,
		prepared_input_data_file,
		id_class_dict_file,
		output_directory,
		output_prefix,
		split_id_dict=None,
		attr_to_split=None,
		attr_to_balance=None,
		gene_balance=None,
		max_trials=100,
		pval_threshold=0.1,
		save_eval_output=True,
		predict_eval=True,
		predict_trainer=False,
		n_hyperopt_trials=0,
		save_gene_split_datasets=True,
		debug_gene_split_datasets=False,
	):
		self.validate_options()

		# load numerical id
		if self.cust_id_class_dict is not None:
			print("Entered here")
			id_class_dict = self.cust_id_class_dict
		else:
			print("Entered here2")
			id_class_dict = pu.load_and_filter(
				None,
				self.nproc,
				id_class_dict_file,
			)

		# load previously filtered and prepared data
		data = pu.load_and_filter(None, self.nproc, prepared_input_data_file)
		data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data

		# define output directory path
		current_date = datetime.datetime.now()
		datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
		if output_directory[-1:] != "/":  # add slash for dir if not present
			output_directory = output_directory + "/"
		output_dir = f"{output_directory}{datestamp}_geneformer_{self.classifier}Classifier_{output_prefix}/"
		subprocess.call(f"mkdir {output_dir}", shell=True)

		# get number of classes for classifier
		num_classes = cu.get_num_classes(self.classifier, self.gene_class_dict)

		if self.classifier == "gene":
			if (gene_balance is True) and (len(self.gene_class_dict.values()) != 2):
				logger.error(
					"Automatically balancing gene sets for training is only available for binary gene classifications."
				)
				raise

			(
				train_list,
				val_list,
				test_list,
				gene_class_keys,
				class_gene_dict,
			) = cu.split_gene_sets(
				self.gene_class_dict,
				self.split_sizes,
				self.num_crossval_splits,
				self.seed,
				save_gene_split_datasets,
				debug_gene_split_datasets,
			)

			id_class_dict = cu.create_gene_id_class_dict(gene_class_keys)
			num_classes = cu.get_num_classes(self.classifier, self.gene_class_dict)

			balance = (gene_balance is True) and (len(self.gene_class_dict.values()) == 2)
			data = cu.prepare_gene_classifier_datasets(
				data,
				train_list,
				val_list,
				test_list,
				gene_class_keys,
				class_gene_dict,
				id_class_dict,
				self.max_cells_per_class,
				balance,
			)

		elif self.classifier == "cell":
			data = cu.prepare_cell_classifier_datasets(
				data,
				id_class_dict,
				self.split_sizes,
				self.num_crossval_splits,
				self.stratify_splits_col,
				split_id_dict,
				attr_to_split,
				attr_to_balance,
				save_gene_split_datasets,
				debug_gene_split_datasets,
			)

		# save id_class_dict
		if self.cust_id_class_dict is None:
			id_class_output_path = (
				Path(output_dir) / f"{output_prefix}_id_class_dict"
			).with_suffix(".pkl")
			with open(id_class_output_path, "wb") as f:
				pickle.dump(id_class_dict, f)

		# run hyperparameter optimization
		if n_hyperopt_trials > 0:
			trainer, eval_results, best_trial = self.hyperopt_classifier(
				model_directory,
				num_classes,
				data["train"],
				data["eval"],
				output_dir,
				n_trials=n_hyperopt_trials,
			)
		else:
			trainer, eval_results, best_trial = self.train_classifier(
				model_directory,
				num_classes,
				data["train"],
				data["eval"],
				output_dir,
			)

		# evaluation
		logger.info("Starting evaluation.")
		if self.num_crossval_splits > 1:
			# generate cross-validated ROC metrics
			logger.info("Calculating cross-validated ROC metrics.")
			roc_metrics_dict = eu.get_cross_valid_roc_metrics(
				trainer,
				data["eval"],
				pval_threshold=pval_threshold,
				max_trials=max_trials,
			)
			eval_output_fname = (
				Path(output_dir) / f"{output_prefix}_eval_output_crossval"
			).with_suffix(".pkl")
			eval_results["roc_metrics"] = roc_metrics_dict
		else:
			eval_output_fname = (
				Path(output_dir) / f"{output_prefix}_eval_output"
			).with_suffix(".pkl")
			roc_metrics_dict = None

		if save_eval_output:
			with open(eval_output_fname, "wb") as f:
				pickle.dump(eval_results, f)

		# prediction
		if predict_eval:
			logger.info("Predicting validation set.")
			predictions = trainer.predict(data["eval"])
			pred_output_fname = (
				Path(output_dir) / f"{output_prefix}_eval_predictions"
			).with_suffix(".pkl")
			with open(pred_output_fname, "wb") as f:
				pickle.dump(predictions, f)

		if predict_trainer:
			logger.info("Predicting training set.")
			predictions = trainer.predict(data["train"])
			pred_output_fname = (
				Path(output_dir) / f"{output_prefix}_train_predictions"
			).with_suffix(".pkl")
			with open(pred_output_fname, "wb") as f:
				pickle.dump(predictions, f)

		return trainer, eval_results, best_trial, roc_metrics_dict

	def hyperopt_classifier(
		self,
		model_directory,
		num_classes,
		train_data,
		eval_data,
		output_directory,
		n_trials=100,
	):
		##### Load model and training args #####
		model = pu.load_model(
			self.model_type,
			num_classes,
			model_directory,
			"train",
			quantize=self.quantize,
		)
		def_training_args, def_freeze_layers = cu.get_default_train_args(
			model, self.classifier, train_data
		)
		del model

		if self.training_args is not None:
			def_training_args.update(self.training_args)
		logging_steps = round(
			len(train_data) / def_training_args["per_device_train_batch_size"]
		)
		def_training_args["logging_steps"] = logging_steps

		training_args = TrainingArguments(
			output_dir=output_directory,
			remove_unused_columns=False,
			**def_training_args,
		)

		def model_init():
			return pu.load_model(
				self.model_type,
				num_classes,
				model_directory,
				"train",
				freeze_layers=def_freeze_layers,
				quantize=self.quantize,
				use_bfloat16=training_args.bf16,
			)

		trainer = Trainer(
			model_init=model_init,
			args=training_args,
			train_dataset=train_data,
			eval_dataset=eval_data,
			data_collator=DataCollatorForGeneClassification(
				use_bfloat16=training_args.bf16
			),
		)

		def objective(trial):
			trial_args = {}
			trial_args["learning_rate"] = trial.suggest_float(
				"learning_rate", 1e-5, 5e-4, log=True
			)
			trial_args["per_device_train_batch_size"] = trial.suggest_categorical(
				"per_device_train_batch_size", [8, 16, 32]
			)

			training_args = TrainingArguments(
				output_dir=output_directory,
				remove_unused_columns=False,
				**{**def_training_args, **trial_args},
			)

			trainer.args = training_args
			trainer.train()
			preds = trainer.predict(eval_data)
			predictions = np.argmax(preds.predictions, axis=-1)
			true_labels = preds.label_ids
			macro_f1 = f1_score(true_labels, predictions, average="macro")

			return macro_f1

		study = optuna.create_study(direction="maximize")
		study.optimize(objective, n_trials=n_trials)

		best_trial = study.best_trial
		best_params = best_trial.params
		def_training_args.update(best_params)
		training_args = TrainingArguments(
			output_dir=output_directory,
			remove_unused_columns=False,
			**def_training_args,
		)

		trainer.args = training_args
		trainer.train()
		eval_results = trainer.evaluate()

		return trainer, eval_results, best_trial

	def model_init(self, model_directory, num_classes, freeze_layers):
		"""
		Initialize model for training.
		"""
		return pu.load_model(
			self.model_type,
			num_classes,
			model_directory,
			"train",
			freeze_layers=freeze_layers,
			quantize=self.quantize,
		)

	def train_classifier(
		self,
		model_directory,
		num_classes,
		train_data,
		eval_data,
		output_directory,
	):

		##### Load model and training args #####
		model = pu.load_model(
			self.model_type,
			num_classes,
			model_directory,
			"train",
			quantize=self.quantize,
		)

		def_training_args, def_freeze_layers = cu.get_default_train_args(
			model, self.classifier, train_data
		)

		if self.training_args is not None:
			def_training_args.update(self.training_args)
		logging_steps = round(
			len(train_data) / def_training_args["per_device_train_batch_size"]
		)
		def_training_args["logging_steps"] = logging_steps

		training_args = TrainingArguments(
			output_dir=output_directory,
			remove_unused_columns=False,
			**def_training_args,
		)

		model = self.model_init(model_directory, num_classes, def_freeze_layers)

		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_data,
			eval_dataset=eval_data,
			data_collator=DataCollatorForGeneClassification(
				use_bfloat16=training_args.bf16
			),
		)

		if not self.no_eval:
			eval_results = trainer.evaluate()
		else:
			eval_results = {}

		return trainer, eval_results, None

	def evaluate_model(
		self,
		model,
		num_classes,
		id_class_dict,
		eval_data,
		predict=False,
		output_directory=None,
		output_prefix=None,
	):

		##### Evaluate the model #####
		labels = id_class_dict.keys()
		y_pred, y_true, logits_list = eu.classifier_predict(
			model, self.classifier, eval_data, self.forward_batch_size
		)
		conf_mat, macro_f1, acc, roc_metrics = eu.get_metrics(
			y_pred, y_true, logits_list, num_classes, labels
		)
		if predict is True:
			pred_dict = {
				"pred_ids": y_pred,
				"label_ids": y_true,
				"predictions": logits_list,
			}
			pred_dict_output_path = (
				Path(output_directory) / f"{output_prefix}_pred_dict"
			).with_suffix(".pkl")
			with open(pred_dict_output_path, "wb") as f:
				pickle.dump(pred_dict, f)
		return {
			"conf_mat": conf_mat,
			"macro_f1": macro_f1,
			"acc": acc,
			"roc_metrics": roc_metrics,
		}

