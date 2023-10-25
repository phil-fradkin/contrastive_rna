#!/bin/bash
#SBATCH --job-name=hl_fine_tune
#SBATCH --output=slurm_out/array_hl_fine_tune_slurm.%A_%a.out
#SBATCH --error=slurm_out/array_hl_fine_tune_slurm.%A_%a.err

# --partition=t4v2,rtx6000,a40
# --account=deadline
# --qos=deadline


#SBATCH --partition=t4v2,rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

#SBATC --partition:t4v2
# --gres=gpu:1
#SBATC --qos=high



#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

#SBATCH -N 1
#SBATCH --ntasks=1

#SBATCH --export=ALL
#SBATCH --array=0,2,4

# prepare your environment here
echo `date`: Job $SLURM_JOB_ID is allocated resource
echo "Starting task $SLURM_ARRAY_TASK_ID"

# virtual env
eval "$(conda shell.bash hook)"
conda activate rna_contrast_2

cd /h/phil/Documents/01_projects/rna_half_life_branch/contrastive_rna_representation/contrastive_rna_representation

# put your command here
python rna_half_life_trainer.py \
--rand_seed ${SLURM_ARRAY_TASK_ID} \
--fraction_of_train 1.0 \
--hl_dataset_parent_dir /scratch/hdd001/home/phil/rna_contrast/datasets/deeplearning/train_wosc/f${SLURM_ARRAY_TASK_ID}_c${SLURM_ARRAY_TASK_ID} \
--resnet less_dilated_small2 \
--global_hl_batch_size 128 \
--dropout_prob 0.3 \
--number_epochs 100 \
--lr 0.01 \
--weight_decay 0 \
--norm_type batchnorm_small_momentum \
--clipnorm 0.5 \
--n_tracks 6 \
--l2_scale_weight_decay 0 \
--lr_decay=false \
--train_full_model=true \
--lr_schedule='exponential' \
--exponential_decay_rate 0.85 \
--kernel_size 2 \
--note 14_wd3-5_9.4_fc \
--fc_head 'fc' \
--save_model=false \
--pooling_layer='avgpool' \
--mixed_precision=true \
--optimizer adam \
--contrastive_checkpoint_epoch 600 \
--contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-9.4_graft_wd3e-5-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_600

# 9.2 graft

# small
# rna_contrast_5-9.2_1e-5wd_graft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_700
# rna_contrast_5-9.2_3e-5wd_graft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_700
# rna_contrast_5-9.2_1e-4wd_graft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_700

# medium
# rna_contrast_5-9.2_1e-5wd_graft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.005-e_700
# rna_contrast_5-9.2_5e-5wd_graft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.005-e_700
# rna_contrast_5-9.2_graft_k4-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.005-e_500

# 9.1

# Less dilated smalll
# rna_contrast_5-9.1_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.05-e_500
# rna_contrast_5-9.1_adamw_wd1e-6-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_500

#4 trk
# rna_contrast_5-9.1_adamw_wd1e-6_4trck-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_500
# rna_contrast_5-9.1_adamw_wd1e-6-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_small2-lr_0.005-e_500


# Less dilated medium
# rna_contrast_5-9.1_adamw_wd1e-6-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.005-e_500
# rna_contrast_5-9.1_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.05-e_500
# rna_contrast_5-9.0_adamw_k4-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.02-e_500


# 9.0
# rna_contrast_5-9.0_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.01-e_1000
# rna_contrast_5-9.0_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_1000

# 8.2 sanity checks
# rna_contrast_5-8.2_adamw_wd1e-6-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_100

# 8.2 try out bigger models
# rna_contrast_5-8.2_adamw_epoch_sched-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_256-less_dilated_medium2-lr_0.01-e_100
# rna_contrast_5-8.2_adamw_k4_wd1e-6-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.05-e_100
# rna_contrast_5-8.1_adamw_epoch_sched_k4-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.01-e_100
# rna_contrast_5-8.1_adamw_epoch_sched_k4-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.01-e_100
# rna_contrast_5-8.2_adamw_epoch_sched_k4-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.05-e_100
# rna_contrast_5-8.2_adamw_epoch_sched_k4-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_100

# Try and find Best runs 8.1
# rna_contrast_5-8.0_adamw_sched_lr_temp-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.05-e_100
# rna_contrast_5-8.0_adamw_sched_lr_temp-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.05-e_100
# rna_contrast_5-8.1_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_100


# rna_contrast_5-8.0_adamw_sched_lr-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.05-e_100
# rna_contrast_5-8.0_adamw_sched_lr-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-less_dilated_medium-lr_0.05-e_100


# Gen 8 bs 1536 smaller dilation higher LR experiment with dropout and new LR schedule

# rna_contrast_5-8.0_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_100/
# rna_contrast_5-8.0_adamw-pool_avgpool-DCLdev_4-d_0.3-seed_0-bs_1536-less_dilated_small-lr_0.01-e_100

# rna_contrast_5-8.0_adamw_lr_sched-pool_avgpool-DCLdev_4-d_0.3-seed_0-bs_1536-less_dilated_small2-lr_0.05-e_100
# rna_contrast_5-8.0_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small-lr_0.05-e_100

# rna_contrast_5-8.0_adamw-pool_avgpool-DCLdev_4-d_0.3-seed_0-bs_1536-less_dilated_small-lr_0.1-e_100
# rna_contrast_5-8.0_adamw-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small-lr_0.1-e_100


#Gen 8
# rna_contrast_5-8.0_adamw_shft_10g-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100
# rna_contrast_5-8.0_adamw_shft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100
# rna_contrast_5-8.0_adamw_shft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-less_dilated_small-lr_0.05-e_100
# rna_contrast_5-8.0_adamw_shft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.05-e_100


# Generation 7

# --contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_alws_dift-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_512-not_dilated_medium-lr_0.005-e_300

# --contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.01-e_1000

# --contrastive_checkpoint_epoch 396
# --contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_b1024_h256-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.01-e_1000

# --contrastive_checkpoint_epoch 536
# --contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_b1024_h256-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_1000




# Generation 4
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-4.0_11g_ds_b2048_h128_tw-pool_avgpool-DCLdev_4-d_0.3-seed_0-bs_512-dilated_medium-lr_0.01-e_1000"


# medium runs

# --contrastive_checkpoint_epoch 1088
# --contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_drop_single_e1000_all_e1200-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.01-e_1200

# --contrastive_checkpoint_epoch 852
# --contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_drop_single_then_all-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.01-e_1000

# --contrastive_checkpoint_epoch 992
# --contrastive_run_dir /scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_drop_single-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.01-e_1000

# Generation 6
# Ablation experiments

# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw_dcl-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"

# Masking ablation | TW 0.2
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw_dcl_no_mask-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw_dcl_small_mask_both-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw_mask_both_.3-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"


# Aumgentation ablation | TW 0.2 | 1-.3-mask
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw_dcl_no_mask-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_only_mask-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_dcl_no_hmlg-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"


# Aumgentation ablation | TW 0.2 | 2-.15-mask
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw_dcl_no_mask-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_only_mask_both_15-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_no_hmlg_both_.15-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_no_aug-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"


# Loss ablation | 1-.3-mask |
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_dcl-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_ntx-pool_avgpool-NTXentLoss_2dev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw1.0_dcl-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw0.6_dcl-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_tw.4_dcl-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100"



# Optimizers

# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_both_.15_adamw_1e-5wd-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-6.0_abl_both_.15_adamw_5e-4wd-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1024-dilated_small-lr_0.01-e_100



# Generation 4

# k=2 dilated_medium big projector
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-5.0_11g_b2048_h512_wd1e-5_bn_mm-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_512-dilated_medium-lr_0.001-e_1000"

# k=2 dilated_medium normal projector
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-5.0_11g_b512_h128_wd1e-5_bn_mm-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_512-dilated_medium-lr_0.001-e_1000"

# k=2 dilated_medium normal projector lamb
# --contrastive_checkpoint_epoch 544
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-5.0_11g_b512_h128_wd1e-5_bn_mm_lamb-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_512-dilated_medium-lr_0.001-e_1000"

# k=4 dilated_medium normal projector adam
# --contrastive_checkpoint_epoch 120 \
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-4.0_11g_k_4_b2048_h128_tw-pool_avgpool-DCLdev_4-d_0.3-seed_0-bs_512-dilated_medium-lr_0.01-e_1000"
