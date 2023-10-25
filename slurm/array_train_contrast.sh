#!/bin/bash
#SBATCH --job-name=cont_9.1
#SBATCH --output=slurm_out/array_gpu_train_contrast_model_slurm.%A_%a.out
#SBATCH --error=slurm_out/array_gpu_train_contrast_model_slurm.%A_%a.err

# --partition=p100,t4v1,t4v2,rtx6000,a40
#SBATCH --partition=a40
#SBATCH --gres=gpu:4
#SBATC --qos=normal
#SBATCH --exclude=gpu052

#SBATCH --account=deadline
#SBATCH --qos=deadline

#SBATCH --cpus-per-task=7
#SBATCH --mem=160G

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --array=0-0


# prepare your environment here
echo `date`: Job $SLURM_JOB_ID is allocated resource
echo "Starting task $SLURM_ARRAY_TASK_ID"

# virtual env
eval "$(conda shell.bash hook)"
conda activate rna_contrast_2

# put your command gene_pair_human_mouse_6t_homolog_drop_single

python ../contrastive_rna_representation/contrastive_model.py \
--rand_seed ${SLURM_ARRAY_TASK_ID} \
--resnet "less_dilated_medium" \
--global_batch_size 768 \
--note "9.1_adamw_wd1e-6" \
--mixed_precision=true \
--lr 0.005 \
--number_epochs 500 \
--proportion_to_mask 0.15 \
--mask_single_transcript=false \
--norm_type syncbatchnorm_small_momentum \
--n_tracks 6 \
--temperature .1 \
--l2_scale_weight_decay 0 \
--kernel_size 2 \
--weight_decay 1e-6 \
--projection_head_size 512 \
--projection_body 2048 \
--contrastive_loss_name DCL \
--pooling_layer avgpool \
--dropout_prob 0.1 \
--optimizer='adamw' \
--clipnorm 1.0 \
--always_different_transcripts=true \
--train_weighted="by_transcript_0.4" \
--dataset_path "$HOME/Documents/01_projects/rna_half_life_branch/contrastive_rna_representation/data_new/10_genome_homologene"

# --dataset_path "$HOME/Documents/01_projects/rna_half_life_branch/contrastive_rna_representation/data_new/11_genome_gene_pair"
# --dataset_path "$HOME/Documents/01_projects/contrastive_rna_representation/data_new/gene_pair_human_mouse_6t_homolog"
# --dataset_path "$HOME/Documents/01_projects/contrastive_rna_representation/data_new/gene_pair_human_6t_homolog"


# losses: NTXentLoss_1, NTXentLoss_2, DCL
# --dataset_path "$HOME/Documents/01_projects/contrastive_rna_representation/data_new/gene_pair_human_mouse_6t_homolog_drop_single"
# --train_weighted="by_transcript" \
