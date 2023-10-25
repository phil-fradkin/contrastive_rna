#!/bin/bash
#SBATCH --job-name=mrl_fine_tune
#SBATCH --output=slurm_out/mrl_fine_tune.%A_%a.out
#SBATCH --error=slurm_out/mrl_fine_tune.%A_%a.err


#SBATCH --partition=rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

#SBATC --account=deadline
#SBATC --qos=deadline


#SBATCH --cpus-per-task=4
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

# DCL - single run

# put your command here
python ../contrastive_rna_representation/mrl_isoform_resolved_trainer.py \
--rand_seed ${SLURM_ARRAY_TASK_ID} \
--resnet less_dilated_small2 \
--global_hl_batch_size 128 \
--dropout_prob 0.3 \
--number_epochs 250 \
--lr 0.01 \
--weight_decay 0. \
--norm_type batchnorm_small_momentum \
--fraction_of_train 1. \
--clipnorm 0.5 \
--n_tracks 6 \
--l2_scale_weight_decay 0 \
--pooling_layer avgpool \
--train_full_model=true \
--lr_schedule='exponential' \
--exponential_decay_rate 0.9 \
--kernel_size 2 \
--note 10_3e-5wd \
--load_pool_weights=false \
--fc_head 'fc_l2' \
--contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs//rna_contrast_5-9.4_graft_wd3e-5-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_600"

# rna_contrast_5-9.1_adamw_wd1e-6-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_500
# rna_contrast_5-9.2_1e-4wd_graft-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_700

# --fc_head 'linear' \

# generation 7
# --contrastive_checkpoint_epoch 724 \
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_alws_dift-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_1536-dilated_small-lr_0.005-e_1000"

# --contrastive_checkpoint_epoch 496 \
# --contrastive_run_dir "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-7.0_alws_dift-pool_avgpool-DCLdev_4-d_0.1-seed_0-bs_768-dilated_medium-lr_0.005-e_1000"


# ${directories[${SLURM_ARRAY_TASK_ID}]:74:19}
# --note 5.0_${SLURM_ARRAY_TASK_ID}_${directories[${SLURM_ARRAY_TASK_ID}]:48:19} \

# Experimenting with homologous gene collapsing:
# "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-1.0_gencode_6t_hm_k3_ph256_m.3-dev_4-d_0.1-seed_0-bs_1024-resnet_dilated_small-lr_0.001-e_1000/"
# "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-1.0_gencode_6t_hm_k3_ph256_m.3_homolog_all-dev_4-d_0.1-seed_0-bs_1024-resnet_dilated_small-lr_0.001-e_1000/"



# Experimenting with projection head and kernel size
# "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-1.0_gencode_6t_hm_k3_ph256-dev_4-d_0.1-seed_0-bs_1024-resnet_dilated_small-lr_0.001-e_1000/"
# "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-1.0_gencode_6t_hm_k8_ph128-dev_4-d_0.1-seed_0-bs_1024-resnet_dilated_small-lr_0.001-e_1000/"
# "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-1.0_gencode_6t_hm_k8_ph256-dev_4-d_0.1-seed_0-bs_1024-resnet_dilated_small-lr_0.001-e_1000/"
# "/scratch/hdd001/home/phil/rna_contrast/runs/rna_contrast_5-1.0_gencode_6t_hm-dev_4-d_0.1-seed_0-bs_1024-resnet_dilated_small-lr_0.001-e_1000/"
