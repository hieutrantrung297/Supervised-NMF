#!/bin/bash
#SBATCH --job-name="SNMF"
#SBATCH --partition=research_hi
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --output={{run_path}}/log.log

source activate /home/hieutran/.conda/envs/hieunho

###################################################### INPUT ################################################
Output={{output_path}}

#################################################################################################
Working_dir={{Working_dir}}
Src=${Working_dir}/src

python3 $Src/run.py \
        {{feature_path}} \
        {{meta_path}} \
        {{output_path}} \
        {{nmf_init_mode}} \
        {{loss_type}} \
        {{feature_name}} \
        {{rank}} \
        {{iter}} \
        {{tolerance}} \
        {{patience}} \
        {{alpha}} \
        {{epsilon}}