#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=2
#$ -N  cut_sup_mmwhs  # Specify job name


source activate py37


CUDA_VISIBLE_DEVICES=0,1 python CUTExperiment.py --model cut_sup