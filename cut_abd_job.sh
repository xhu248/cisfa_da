#!/bin/bash

# JSUB -J cut_atten_abd
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
