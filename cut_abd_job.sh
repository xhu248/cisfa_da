#!/bin/bash

dataset=abd
fold=0
train_sample=1
num_classes=5
batch_size=4
netG=smallstylegan2
model=cut_atten_coseg_sum
exp=${model}_${dataset}
l_nce=1
l_gcn=1
l_gan=5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/software/anaconda3/lib/
echo "start running job"

python CUTExperiment.py --model ${model} --batch_size ${batch_size} --n_epochs 200 --seg_start_point 0 \
--n_epochs_decay 0 \
--fold ${fold} \
--num_classes ${num_classes} \
--nce_idt False \
--lambda_NCE ${l_nce} \
--src_dir ../data/abdominal_data/multi_atlas  --src_data_dir ../data/abdominal_data/multi_atlas/cropped \
--target_dir ../data/abdominal_data/chaos --target_data_dir ../data/abdominal_data/chaos/cropped \
--name ${exp}_f${fold}_b${batch_size} \
> output_log/${exp}_f${fold}_b${batch_size}
