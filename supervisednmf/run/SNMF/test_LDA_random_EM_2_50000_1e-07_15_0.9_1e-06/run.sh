#!/bin/bash
#SBATCH --job-name="SNMF"
#SBATCH --partition=research_hi
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --output=/mnt/DATASM14/hieunho/hieu_gitlab/supervisednmf/run/SNMF/test_LDA_random_EM_2_50000_1e-07_15_0.9_1e-06/log.log

source activate /home/hieutran/.conda/envs/hieunho

###################################################### INPUT ################################################
Output=/mnt/DATASM14/hieunho/hieu_gitlab/supervisednmf/output/SNMF/test_LDA_random_EM_2_50000_1e-07_15_0.9_1e-06

#################################################################################################
Working_dir=/mnt/DATASM14/hieunho/hieu_gitlab/supervisednmf/
Src=${Working_dir}/src

python3 $Src/run.py \
        /mnt/DATASM14/hieunho/hieu_gitlab/supervisednmf/feature/EM/feature.csv \
        /mnt/DATASM14/hieunho/hieu_gitlab/supervisednmf/meta/EM/meta.csv \
        /mnt/DATASM14/hieunho/hieu_gitlab/supervisednmf/output/SNMF/test_LDA_random_EM_2_50000_1e-07_15_0.9_1e-06 \
        random \
        LDA \
        EM \
        2 \
        50000 \
        1e-07 \
        15 \
        0.9 \
        1e-06