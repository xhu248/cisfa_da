#!/bin/bash

# JSUB -J cut_sup_seg_abd
# JSUB -q tensorflow_sub
# JSUB -gpgpu "num=2"
# JSUB -R "span[ptile=2]"
# JSUB -n 2
# JSUB -o ./log/output.%J
# JSUB -e ./log/err.%J

##########################Cluster variable######################
if [ -z "$LSB_HOSTS" -a -n "$JH_HOSTS" ]
then

        for var in ${JH_HOSTS}
        do
                if ((++i%2==1))
                then
                        hostnode="${var}"
                else
                        ncpu="$(($ncpu + $var))"
                        hostlist="$hostlist $(for node in $(seq 1 $var);do printf "%s " $hostnode;done)"
                fi
        done
        export LSB_MCPU_HOSTS="$JH_HOSTS"
        export LSB_HOSTS="$(echo $hostlist|tr ' ' '\n')"
fi

nodelist=.hostfile.$$
for i in `echo $LSB_HOSTS`j
do
    echo "${i}" >> $nodelist
done

ncpu=`echo $LSB_HOSTS |wc -w`

##########################Software environment variable#####################
module load cuda/cuda10.1
module load anaconda/anaconda3-python3.7
source activate py37

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
--src_dir ../data/sifa_data/mri  --src_data_dir ../data/sifa_data/mri/cropped \
--target_dir ../data/sifa_data/ct --name ${model}_${dataset}_b${batch_size} \
--target_data_dir ../data/sifa_data/ct/cropped
