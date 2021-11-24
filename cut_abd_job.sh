#!/bin/bash

dataset=abd
fold=0
train_sample=1
num_classes=5
batch_size=4
netG=smallstylegan2
model=cut_atten_coseg_sum
exp=${model}_${dataset}
l_pcl=1
l_gcl=1
l_gan=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/software/anaconda3/lib/
echo "start running job"

python CUTExperiment.py --model ${model} --batch_size ${batch_size} --n_epochs 200 --seg_start_point 0 \
--n_epochs_decay 0 \
--fold ${fold} \
--num_classes ${num_classes} \
--pcl_idt False \
--lambda_PCL ${l_pcl} \
--src_dir ${where_you_store_the_source_data}  --src_data_dir ${the_direct_you_store_the_processed_source_data} \
--target_dir ${where_you_store_the_target_data} --target_data_dir ${where_you_store_the_processed_target_data} \
--name ${exp}_f${fold}_b${batch_size}
# > output_log/${exp}_f${fold}_b${batch_size}
