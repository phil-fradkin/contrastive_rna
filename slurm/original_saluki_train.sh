#!/bin/bash
#SBATCH --job-name=saluki
#SBATCH --output=slurm_out/saluki%A_%a.out
#SBATCH --error=slurm_out/saluki.%A_%a.err

#SBATCH --partition=t4v2,rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

# --partition=t4v2
# --gres=gpu:1
# --qos=high

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

#SBATCH -N 1
#SBATCH --ntasks=1

#SBATCH --export=ALL
#SBATCH --array=0

# prepare your environment here
echo `date`: Job $SLURM_JOB_ID is allocated resource
echo "Starting task $SLURM_ARRAY_TASK_ID"

# virtual env
eval "$(conda shell.bash hook)"
conda activate rna_contrast_2

fraction_of_train=(
    1.0
    0.1
    0.01
    0.5
    0.3
    0.005
)

RAND_SEED=0
# put your command here
python ../contrastive_rna_representation/rna_half_life_trainer.py \
--rand_seed ${RAND_SEED} \
--hl_dataset_parent_dir /scratch/hdd001/home/phil/rna_contrast/datasets/deeplearning/train_wosc/f${RAND_SEED}_c${RAND_SEED} \
--resnet "saluki" \
--global_hl_batch_size 64 \
--dropout_prob 0.3 \
--number_epochs 2400 \
--lr 0.0001 \
--norm_type "batchnorm_small_momentum" \
--clipnorm 0.5 \
--save_model=false \
--n_tracks 6 \
--note 11.0_saluki \
--fraction_of_train ${fraction_of_train[${SLURM_ARRAY_TASK_ID}]} \
--l2_scale_weight_decay 0.001 \
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-8.0_saluki-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_2048-saluki-lr_0.001-e_100/"
