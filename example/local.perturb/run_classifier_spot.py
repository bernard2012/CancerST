import numpy as np
import os
import datetime
from new_classifier import Classifier
from datasets import load_from_disk
import pandas as pd
from itertools import product
import sys
import random

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
	return opposite_race_pairs, all_patients_list, african_american, white

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

ray_config = {"num_train_epochs": [5,], #default 10
"learning_rate": (2e-4, 1e-3), #default 1e-6, 1e-3
"weight_decay": (0.25, 0.35), #default 0-0.35 (0.25-0.35)
"lr_scheduler_type": ["linear", "cosine", "cosine_with_restarts"],
"warmup_steps": (100, 1000), #default 100-2000 (for batch size 12) (it was 1000-2000)
"seed": (0, 100),
#"label_smoothing_factor": [0.0, 0.05, 0.1],
"per_device_train_batch_size": [12,], #if it is 36, then divide warmup steps by 3
}

# get all possible withold pairs
exclude_list = []  # List of excluded Patient IDs, if any
withhold_sets, all_ids, african_american, white = generate_opposite_race_pairs(exclude_list)
AA = african_american["Patient ID"].tolist()
EA = white["Patient ID"].tolist()

uniq_AA = list(set(AA))
uniq_EA = list(set(EA))

random.seed(700) #200

random.shuffle(uniq_AA)
random.shuffle(uniq_EA)

print(uniq_AA)
print(uniq_EA)

print(len(uniq_AA))
print(len(uniq_EA))

EA_train = uniq_EA[0:5]
EA_eval = uniq_EA[5:6]
EA_test = uniq_EA[6:]

AA_train = uniq_AA[0:13]
AA_eval = uniq_AA[13:14]
AA_test = uniq_AA[14:]

print(len(EA_train), len(EA_eval), len(EA_test))
print(len(AA_train), len(AA_eval), len(AA_test))

split_id_dict = {"attr_key": "Patient", "train": EA_train + AA_train + EA_eval + AA_eval, "test": AA_test + EA_test}
split_id_dict_2 = {"attr_key": "Patient", "train": EA_train + AA_train, "eval": EA_eval + AA_eval}

data = load_from_disk("/media/scandisk/RacialTNBC/STGeneformer_GSE210616_Unfiltered.dataset")
id_class_dict = {1: "African-American", 0: "Non-Hispanic White"}

cc = Classifier(classifier="spot", spot_state_dict = {"state_key": "Race", "states": "all"}, filter_data=None,
training_args=training_args, max_num_spots=10_000, freeze_layers = 4, num_crossval_splits = 1, #max_num_spots is None previously
forward_batch_size=200, nproc=32, ray_config=ray_config, cust_id_class_dict = id_class_dict)

output_dir = f"/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/{datestamp}"
output_prefix = f"{datestamp}"
os.makedirs(output_dir, exist_ok=True)

cc.prepare_data("/media/scandisk/RacialTNBC/STGeneformer_GSE210616_Unfiltered.dataset", output_dir, output_prefix, 
split_id_dict=split_id_dict)

all_metrics = cc.validate(model_directory="/media/scandisk/Perturbation/models",
prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl", o_directory=output_dir, o_prefix=output_prefix, 
split_id_dict=split_id_dict_2, save_eval_output=True, n_hyperopt_trials=2)
