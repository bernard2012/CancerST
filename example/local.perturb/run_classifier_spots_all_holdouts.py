import os
import datetime
from new_classifier import Classifier
from datasets import load_from_disk
import pandas as pd
from itertools import product
import sys

def generate_opposite_race_pairs(exclude_list):
	# Load the correct table
	df = pd.read_csv("GSE210616_Table.csv")
	# Filter available patients
	available_patients = df[~df['Patient ID'].isin(exclude_list)]
	# Separate African-American and Non-Hispanic White patients
	african_american = available_patients[available_patients['Race'] == 'African-American']
	white = available_patients[available_patients['Race'] == 'Non-Hispanic White']
	# Generate all possible pairs of opposite race
	print("AA", african_american["Patient ID"].tolist())
	print("White", white["Patient ID"].tolist())
	opposite_race_pairs = list(set(product(african_american['Patient ID'].tolist(), white['Patient ID'].tolist())))
	# List of all patients
	all_patients_list = available_patients["Patient ID"].tolist()
	return opposite_race_pairs, all_patients_list

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

# set training arguments
training_args = {
	'learning_rate': 0.00017059048723147853,
	'lr_scheduler_type': 'cosine',
	'warmup_steps': 1187.8023461527375,
	'weight_decay': 0.26712350685637704,
	"per_device_eval_batch_size": 12,
	'num_train_epochs': 1, 
	'seed': 64,
}

# get all possible withold pairs
exclude_list = []  # List of excluded Patient IDs, if any
withhold_sets, all_ids = generate_opposite_race_pairs(exclude_list)
print(len(withhold_sets))
finished_sets = [(1,8),(1,15),(3,6),(3,8),(3,13),(3,15),(9,5),(9,8),(9,14),(11,5),(11,7),(11,14),(12,13),(17,5),(17,7),\
(17,14),(18,5),(18,14),(19,6),(20,8),(20,15),(22,5),(22,14),(23,13),(24,5),(24,8),(24,14),
#(18,7)
]

print("Withhold sets")
print(withhold_sets)

print(all_ids)

#sys.exit(0)

withhold_sets = [tup for tup in withhold_sets if tup not in finished_sets]
print(len(withhold_sets))

for ws in withhold_sets:
	output_prefix = f"Patient_{ws[0]}_and_{ws[1]}_withheld"
	output_dir = f"Race_Classifier_All_Holdouts/{output_prefix}"
	os.makedirs(output_dir, exist_ok=True)

	cc = Classifier(classifier="spot", spot_state_dict = {"state_key": "Race", "states": "all"}, filter_data=None,
	training_args=training_args, max_num_spots=None, freeze_layers = 0, num_crossval_splits = 1, eval_size= 0.2,
	stratify_splits_col = "Race", forward_batch_size=200, nproc=16)
	
	print(f"Witholding patients {ws[0]} and {ws[1]}")
	
	train_test_id_split_dict = {"attr_key": "Patient", "train": list(set(all_ids)-set(ws)), "test": list(ws)}
	
	cc.prepare_data("/media/scandisk/RacialTNBC/STGeneformer_GSE210616_Unfiltered.dataset", output_dir, output_prefix, split_id_dict=train_test_id_split_dict)

	all_metrics = cc.validate(model_directory="/media/scandisk/Perturbation/models/", prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
	id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl", o_directory=output_dir, o_prefix=output_prefix, save_eval_output=True)
	
	cc = Classifier(classifier="spot", spot_state_dict = {"state_key": "Race", "states": "all"}, forward_batch_size=200, nproc=16)
	
	all_metrics_test = cc.evaluate_saved_model(
			model_directory=f"{output_dir}/geneformer_spotClassifier_{output_prefix}/ksplit1/",
			id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
			test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
			o_directory=output_dir,
			o_prefix=output_prefix,
		)
	
	cc.plot_conf_mat(
			conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
			o_directory=output_dir,
			o_prefix=output_prefix,
	)
	
	cc.plot_predictions(
		predictions_file=f"{output_dir}/{output_prefix}_pred_dict.pkl",
		id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
		title="Race",
		o_directory=output_dir,
		o_prefix=output_prefix,
	)
	
	cc.plot_roc(all_metrics_test["all_roc_metrics"], {'color': 'black', 'linestyle': '-'}, "ROC_Plot", output_dir, output_prefix)

