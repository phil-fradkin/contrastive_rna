#!/bin/bash
#SBATCH --job-name=go
#SBATCH --output=slurm_out/go_slurm.%A_%a.out
#SBATCH --error=slurm_out/go_slurm.%A_%a.err

##t4v2,rtx6000,a40
#--partition=t4v2
#--gres=gpu:1
#--qos=high


#SBATCH --partition=rtx6000,t4v2
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

#SBATCH -N 1
#SBATCH --ntasks=1

#SBATCH --export=ALL
#SBATCH --array=0-2

# prepare your environment here
echo `date`: Job $SLURM_JOB_ID is allocated resource
echo "Starting task $SLURM_ARRAY_TASK_ID"

# virtual env
eval "$(conda shell.bash hook)"
conda activate rna_contrast_2

# put your command here
python ../contrastive_rna_representation/go_train.py \
--note "10.0_sml_frz_k2_e300" \
--rand_seed ${SLURM_ARRAY_TASK_ID} \
--resnet "less_dilated_small2" \
--global_batch_size 128 \
--dropout_prob 0.3 \
--number_of_epochs 150 \
--lr 0.0035 \
--norm_type "batchnorm_small_momentum" \
--clipnorm 0.5 \
--n_tracks 4 \
--l2_scale_weight_decay 0 \
--kernel_size 2 \
--single_transcript=true \
--high_evidence=false \
--num_classes 10 \
--train_full_model=false \
--fc_head='linear_sigmoid' \
--contrastive_checkpoint_epoch 500 \
--contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-9.1_adamw_wd1e-6_4trck-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_500

# generation 7
# --contrastive_checkpoint_epoch 648 \
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_alws_dift-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.005-e_650"

# --contrastive_checkpoint_epoch 428 \
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_alws_dift-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.005-e_1000"

# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-4.0_11g_ds_b2048_h128_tw-pool_avgpool-DCLdev_4-d_0.3-seed_0-bs_512-dilated_medium-lr_0.01-e_1000"
# --contrastive_run_dir="/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-3.0_b2048_h128_tw-pool_avgpool-DCLdev_4-d_0.3-seed_0-bs_1024-dilated_small-lr_0.001-e_1000"
