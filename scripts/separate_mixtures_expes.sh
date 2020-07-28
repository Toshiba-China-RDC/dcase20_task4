#!/bin/bash

# Use the baseline source separation model to separate the mixtures from the different datasets
# Weak, unlabel_in_domain, validation and synthetic are separated.

BASE_DATASET=../dataset
AUDIO_PATH_WEAK=${BASE_DATASET}/audio/train/weak
AUDIO_PATH_UNLABEL=${BASE_DATASET}/audio/train/unlabel_in_domain
AUDIO_PATH_VALIDATION=${BASE_DATASET}/audio/validation/validation
AUDIO_PATH_EVAL=${BASE_DATASET}/audio/eval_label/youtube

# Change synthetic20 to synthetic20_reverb if you want to separate the reverbed data
SCRIPTS_PATH=../data_generation

# Download the checkpoint model from FUSS baseline
# If you want to use another model from Google, just change the path of the CHECKPOINT_MODEL and INFERENCE_META
# Todo, download the models from somewhere (shareable by Google team) if not already done
#wget -O FUSS_DESED_baseline_dry_2_model.tar.gz https://zenodo.org/record/3743844/files/FUSS_DESED_baseline_dry_2_model.tar.gz
#tar -xzf FUSS_DESED_baseline_dry_2_model.tar.gz
#rm FUSS_DESED_baseline_dry_2_model.tar.gz
#mv fuss_desed_baseline_dry_2_model ../fuss_desed_baseline_dry_2_model


declare -a models=(dry_1 dry_2 dry_4 dry_4np dry_5 dry_6)

for model_name in "${models[@]}"
do
	checkpoint_model=../ss_model/fuss_desed_${model_name}model
	inference=../ss_model/fuss_desed_${model_name}inference.meta

	# Recorded data
	declare -a arr=(${AUDIO_PATH_WEAK} ${AUDIO_PATH_UNLABEL} ${AUDIO_PATH_VALIDATION} ${AUDIO_PATH_EVAL})

	for audio_path in "${arr[@]}"
	do
	   echo "${audio_path}"
	   python ${SCRIPTS_PATH}/separate_wavs.py --audio_path=${audio_path}16k \
	   --output_folder=${audio_path}_ss/separated_sources_${model_name} --checkpoint=${checkpoint_model} --inference=${inference}
	   # or do whatever with individual element of the array
	done

	for subset in train validation
	do
		audio_synth=${BASE_DATASET}/audio/${subset}/synthetic20_${subset}/soundscapes
		sources_synth=${BASE_DATASET}/audio/${subset}/synthetic20_${subset}/separated_sources_${model_name}
		# Synthetic (generated) data
		python ${SCRIPTS_PATH}/separate_wavs.py --audio_path=${audio_synth} \
		--output_folder=${sources_synth} --checkpoint=${checkpoint_model} --inference=${inference}
	done
done
