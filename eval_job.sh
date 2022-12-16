#!/bin/bash

# dataset=hippo
dataset=abdominal
fold=1
train_sample=1
num_classes=5
exp=dice_mri
batch_size=4
model=cut_atten_coseg_sum

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/software/anaconda3/lib/
echo "start running job"

python evaluate.py --model ${model} --batch_size ${batch_size} \
--fold ${fold} \
--num_classes ${num_classes} \
--src_dir ./data/mmwhs/mri  --src_data_dir ../data/mmwhs/mri/cropped \
--target_dir ../data/mmwhs/ct --name ${model}_${dataset}_b${batch_size} \
--target_data_dir ../data/mmwhs/ct/cropped
