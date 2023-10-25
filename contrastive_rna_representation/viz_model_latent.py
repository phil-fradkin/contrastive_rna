import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os
import umap

from contrastive_rna_representation.contrastive_model import make_or_restore_model
from contrastive_rna_representation.go_train import (
    get_root_children_go_terms,
    get_two_levels_below_root_go_terms,
    get_three_levels_below_root_go_terms
)
from contrastive_rna_representation.util import make_timestamp, load_appris, load_transcript_map
from contrastive_rna_representation.data import RefseqDataset


sns.set()
sns.set_context('talk')


def load_go_terms(go_path='../data/'):
    godf = pd.read_csv(f'{go_path}/gene_ontology.csv')
    vc = godf[['gene_name', 'id']].value_counts()
    print(len(vc.keys()))
    print(pd.Series(vc.values).describe())
    godf['category'].value_counts()
    godf_cc = pd.read_csv(f'{go_path}/gene_ontology_cc.csv')
    vc = godf_cc[['gene_name', 'id']].value_counts()
    print(len(vc.keys()))
    print(pd.Series(vc.values).describe())
    godf_cc['gocategory'].value_counts()
    godf_bp = pd.read_csv(f'{go_path}/gene_ontology_bp.csv')
    vc = godf_bp[['gene_name', 'id']].value_counts()
    print(len(vc.keys()))
    print(pd.Series(vc.values).describe())
    godf_bp['gocategory'].value_counts()
    return godf, godf_cc, godf_bp


def load_go_root_terms(go_tree='cp'):
    root_go_terms = get_root_children_go_terms(go_tree=go_tree)
    two_below_root = get_two_levels_below_root_go_terms(go_tree=go_tree)
    three_below_root = get_three_levels_below_root_go_terms(go_tree=go_tree)

    print(len(three_below_root),len(two_below_root), len(root_go_terms))
    return root_go_terms, two_below_root, three_below_root


def load_contrastive_model(
    model_dir="/scratch/hdd001/home/phil/rna_contrast/runs/",
    model_path=(
        "rna_contrast_5-4.0_11g_b2048_h128_tw-pool_avgpool-"
        "DCLdev_4-d_0.3-seed_0-bs_1024-dilated_small-lr_0.01-e_1000"
    ),
    contrastive_checkpoint=0,
    n_tracks=6,
):
    sample = tf.constant(np.random.randn(3, 12288, n_tracks), dtype=tf.float32)
    model, contrast_epoch = make_or_restore_model(
        f"{model_dir}/{model_path}/checkpoints",
        {
            "lr": 1,
            "hl_lr": 1,
            "mixed_precision": False,
            "weight_decay": 0,
            'l2_scale_weight_decay': 0,
            'clipnorm': 5,
            'checkpoint_epoch': contrastive_checkpoint,
        },
        checkpoint_epoch=contrastive_checkpoint,
    )
    model.compute_output_shape(input_shape=(None, 12288, n_tracks))
    output2 = model(sample)
    model.model.projection_head = None

    print([x.dtype for x in output2])
    print([x.shape for x in output2])

    print(model.model.summary())
    return model




def sample_classes(numeric_labels, min_number=None):
    """_summary_
    takes in a list of numeric labels and returns positions at
    which the labels will be balanced
    Args:
        numeric_labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    numeric_labels = np.array(numeric_labels)
    label_vc = pd.Series(numeric_labels).value_counts()
    if not min_number:
        min_number = label_vc.min()

    sampled_positions_all = []

    for label in label_vc.keys():
        indecies_of_labels = np.where(numeric_labels == label)[0]
        np.random.seed(42)
        np.random.shuffle(indecies_of_labels)
        sampled_positions = indecies_of_labels[:min_number]
        sampled_positions_all.extend(sampled_positions)

    sampled_positions_all = np.array(sampled_positions_all)
    print(
        np.array(sampled_positions_all).shape,
        pd.Series(sampled_positions_all).duplicated().sum()
    )
    return sampled_positions_all


def create_umap_visualization(
    embeddings,
    labels,
    alpha=.1,
    metric='euclidean',
    min_dist=0,
    balance=False,
    cmap='magma',
    title='',
    rev_map=None,
    plot_type='scatter',
    n_neighbors=15
):
    """
    Create a UMAP visualization of the given embeddings in two dimensions.

    Args:
        embeddings (list of numpy arrays): The embeddings to visualize.
        labels (list): The labels corresponding to each embedding.

    Returns:
        None.
    """
    # plt.style.use('dark_background')
    # Use UMAP to transform the embeddings to two dimensions
    reducer = umap.UMAP(
        metric=metric,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)
        # colors = plt.cm.tab20(np.linspace(0, 1, num_labels))

    unique_labels = list(set(labels))
    num_labels = len(unique_labels)
    colors = get_colors(num_labels, cmap)

    if balance:
        sampled_positions = sample_classes(labels)
        embedding_2d = embedding_2d[sampled_positions, :]
        labels = labels[sampled_positions]

        # Create a scatter plot of the embeddings in two dimensions
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if rev_map:
                label_name = rev_map[label]
            else:
                label_name = label

            if 'scatter' in plot_type:
                plt.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c=[colors[i]],
                    label=label_name,
                    alpha=alpha
                )
            if 'kde' in plot_type:
                sns.kdeplot(
                    x=embedding_2d[mask, 0],
                    y=embedding_2d[mask, 1],
                    label=label_name,
                    levels=3,
                    thresh=.2
                )

    else:
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        embedding_2d = embedding_2d[indices, :]
        labels = labels[indices]

        # Create a scatter plot of the embeddings in two dimensions
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if rev_map:
                label_name = rev_map[label]
            else:
                label_name = label

            if 'scatter' in plot_type:
                plt.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c=[colors[i]],
                    label=label_name,
                    alpha=alpha
                )
            if 'kde' in plot_type:
                sns.kdeplot(
                    x=embedding_2d[mask, 0],
                    y=embedding_2d[mask, 1],
                    label=label_name,
                    levels=3,
                    thresh=.2
                )
    plt.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False
    )

    plt.title(f"{title} UMAP")
    plt.legend()

def visualize_pca(X, labels, alpha=.1, cmap='magma', rev_map=None):
    """
    Visualize the first two principal components of a PCA.

    Args:
        X (numpy array): The data to be transformed and visualized.
        labels (numpy array or list, optional): Labels to use for
        coloring the points in the plot. If not provided, all points
        will be plotted in the same color.

    Returns:
        None.
    """
    # plt.style.use('dark_background')

    # Create a PCA object with two components
    pca = PCA(n_components=2)

    # Transform the data using the PCA object
    X_pca = pca.fit_transform(X)

    unique_labels = list(set(labels))
    num_labels = len(unique_labels)
    colors = get_colors(num_labels, cmap)

    # Create a scatter plot with colored points
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if rev_map:
            label_name = rev_map[label]
        else:
            label_name = label

        # Plot the first two principal components, colored by the provided labels (if any)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],c=[colors[i]], label=label_name, alpha=alpha)

    plt.title('First Two Principal Components')

    # Add labels to the axes indicating the percentage of variance explained by each component
    plt.xlabel(f'Principal Component 1 ({round(pca.explained_variance_ratio_[0]*100, 2)}% of variance)')
    plt.ylabel(f'Principal Component 2 ({round(pca.explained_variance_ratio_[1]*100, 2)}% of variance)')



def plot_tsne(embedding, labels, perplexity=30, title='', alpha=.3, rev_map=None, cmap='magma', legend=True):
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        learning_rate='auto',
        n_iter=1000,
        init='pca'
    )
    embedding_tsne = tsne.fit_transform(embedding)

    unique_labels = list(set(labels))
    num_labels = len(unique_labels)

    colors = get_colors(num_labels, cmap)
    # Create a scatter plot with colored points
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if rev_map:
            label_name = rev_map[label]
        else:
            label_name = label

        plt.scatter(
            embedding_tsne[mask, 0],
            embedding_tsne[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=alpha
        )

    # Add a colorbar legend
    if legend:
        plt.legend()
    plt.title(f"{title}")
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)


def get_colors(num_labels, cmap):
    if cmap == 'Set2':
        colors = plt.cm.Set2(np.linspace(.2, .8, num_labels))
    elif cmap == 'magma':
        colors = plt.cm.magma(np.linspace(.13, .8, num_labels))
    elif cmap == 'winter':
        colors = plt.cm.winter(np.linspace(.1, .9, num_labels))
    elif cmap == 'viridis':
        colors = plt.cm.viridis(np.linspace(.1, .9, num_labels))
    return colors


def generate_transcript_dataset_and_labels(df, map):
    t_dataset = []
    labels = []
    pad_length_to = 12288
    zero_mean = False
    zero_pad = True
    n_tracks = 6

    assert 'gene_name' in df.columns
    assert 'Transcript ID' in df.columns
    assert 'term' in df.columns

    df['in_map'] = [x in map.keys() for x in df['gene_name']]
    df = df[df['in_map']]

    for index, row in tqdm(df.iterrows()):
        transcripts = [t for t in map[row['gene_name']] if t.transcript_id.split('.')[0] == row['Transcript ID']]

        if len(transcripts) != 1:
            continue

        t = transcripts[0]
        sample = np.zeros((12288, 6))
        sample[
            :, 0:4
        ] = t.one_hot_encode_transcript(
            pad_length_to=pad_length_to, zero_mean=zero_mean, zero_pad=zero_pad
        )
        if n_tracks >= 5:
            sample[
                :, 4:5
            ] = t.encode_coding_sequence_track(pad_length_to=pad_length_to)
        if n_tracks >= 6:
            sample[
                :, 5:6
            ] = t.encode_splice_track(pad_length_to=pad_length_to)

        t_dataset.append(sample)
        labels.append(row['term'])
    t_dataset = np.stack(t_dataset, axis=0)
    print(t_dataset.shape)
    return t_dataset, labels



def create_embedding_and_labels(df, model, map, inference_method='contrastive'):

    t_dataset, labels = generate_transcript_dataset_and_labels(df, map)
    if inference_method == 'contrastive':
        def inference_method(x):
            return model.model.avgpool(model.model.representation(tf.constant(x)))
    elif inference_method == 'supervised':
        def inference_method(x):
            return model.avgpool(model.representation(tf.constant(x)))
    else:
        raise ValueError
    # inference_method = lambda x: model.model.avgpool(model(tf.constant(x))[2])
    # inference_method = lambda x: model(tf.constant(x))[1]

    batch_size = 512
    n_batches = t_dataset.shape[0] // batch_size + 1
    embeddings = []

    for batch_num in range(n_batches):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size
        batch = t_dataset[start: end, :, : ]
        output = inference_method(batch)
        embeddings.extend(output)

    # enum_map = {x: i for i, x in enumerate(set(labels))}
    manual_rename_dict = {
        'G protein-coupled receptor signaling pathway': 'G protein-coupled receptor',
        'DNA-binding transcription factor activity, RNA polymerase II-specific': 'DNA-binding transcription factor'

    }
    reverse_map = dict()
    enum_map = dict()
    for i, x in enumerate(set(labels)):
        enum_map[x] = i

        if x in manual_rename_dict.keys():
            x = manual_rename_dict[x]

        reverse_map[i] = x

    numeric_labels = np.array([enum_map[x] for x in labels])
    print(len(embeddings), len(labels), pd.Series(labels).value_counts())
    embeddings = np.array(embeddings)
    numeric_labels = np.array(numeric_labels)
    return embeddings, numeric_labels, reverse_map


## Visualize Cellular Components
def preprocess_data(godf, n_below_root, n_go_terms, app_h, drop_multi_go='all'):
    go_data = godf
    go_term_subset = n_below_root

    print(f"Number of unique genes: {len(go_data['gene_name'].unique())}")
    # subset to root level go terms
    gene_root_go = go_data[go_data['id'].isin([x[0] for x in go_term_subset])]
    print(f"Number of genes with root level GO {gene_root_go['gene_name'].nunique()}")
    gene_root_go = gene_root_go[~gene_root_go[['gene_name', 'id']].duplicated()]

    top_10_go_terms = gene_root_go['id'].value_counts()[:n_go_terms].keys()

    print(gene_root_go[['id', 'term']].value_counts()[:n_go_terms])
    gene_root_go = gene_root_go[gene_root_go['id'].isin(top_10_go_terms)]

    print(gene_root_go.head().to_string())
    gene_root_go.value_counts(['id', 'term'])

    if drop_multi_go == 'all':
        gene_root_go = gene_root_go[~gene_root_go['gene_name'].duplicated(keep=False)]
    elif drop_multi_go == 'first':
        gene_root_go = gene_root_go[~gene_root_go['gene_name'].duplicated(keep='first')]
    else:
        raise ValueError

    gene_root_go = app_h.merge(
        gene_root_go, left_on='Gene name', right_on='gene_name'
    )[['gene_name', 'Gene ID', 'Transcript ID', 'id', 'evidence', 'term']]
    print(len(gene_root_go))

    return gene_root_go


def generate_embed_from_gene_root_go(
    gene_root_go,
    model,
    map,
    number_of_sample_positions=300
):

    embeddings1, numeric_labels1, rev_map1 = create_embedding_and_labels(
        gene_root_go, model, map
    )
    if number_of_sample_positions:
        # sample random positions
        temp_sampled_positions1 = sample_classes(
            numeric_labels1,
            min_number=number_of_sample_positions
        )
        temp_embeddings1 = embeddings1[temp_sampled_positions1, :]
        temp_numeric_labels1 = numeric_labels1[temp_sampled_positions1]
        temp_random_labels1 = temp_numeric_labels1.copy()
        np.random.shuffle(temp_random_labels1)
    else:
        temp_embeddings1 = embeddings1
        temp_numeric_labels1 = numeric_labels1
        temp_random_labels1 = temp_numeric_labels1.copy()
        np.random.shuffle(temp_random_labels1)

    return temp_embeddings1, temp_numeric_labels1, temp_random_labels1, rev_map1


def create_viz_for_model_and_go_tree(
    model,
    godf,
    app_h,
    map,
    root_go_term,
    title='Molecular Function',
    n_go_terms=3,
    n_down_frm_root=3,
    gene_root_go=(),
    number_of_sample_positions=300,
    directory_name='plots',
):
    if not any(gene_root_go):
        gene_root_go = preprocess_data(
            godf=godf,
            n_below_root=root_go_term,
            n_go_terms=n_go_terms,
            app_h=app_h
        )
    temp_embeddings1, temp_numeric_labels1, temp_random_labels1, rev_map1 = generate_embed_from_gene_root_go(
        gene_root_go,
        model,
        map,
        number_of_sample_positions=number_of_sample_positions
    )
    print(
        len(temp_embeddings1),
        len(temp_numeric_labels1),
        pd.Series(temp_numeric_labels1).value_counts()
    )
    figsize = (7, 7)
    save_name = "".join(title.split(" "))

    save_path = f"../plots/embedding_viz_2/{directory_name}"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for perplexity in [
        10, 30, 40,
        50, 60, 80
    ]:
        plt.figure(figsize=figsize)
        plot_tsne(
            temp_embeddings1,
            temp_numeric_labels1,
            perplexity=perplexity,
            title=" ".join(title.split("_")),
            alpha=.7,
            rev_map=rev_map1,
            cmap='magma'
        )
        plt.tight_layout()
        plt.savefig(
            f'{save_path}/{save_name}_t_sne_top_{n_go_terms}_go_root_terms_{perplexity}.pdf'
        )


def create_viz_for_model_by_gene(
    model,
    sampled_by_gene,
    map,
    title='',
    directory_name='plots',
):
    temp_embeddings1, temp_numeric_labels1, temp_random_labels1, rev_map1 = generate_embed_from_gene_root_go(
        sampled_by_gene,
        model,
        map,
        number_of_sample_positions=0
    )
    print(
        len(temp_embeddings1),
        len(temp_numeric_labels1),
        pd.Series(temp_numeric_labels1).value_counts()
    )
    figsize = (7, 7)
    save_name = "".join(title.split(" "))

    save_path = f"../plots/embedding_viz_2/{directory_name}"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for perplexity in [
        10, 30, 40,
        50, 60, 80
    ]:
        plt.figure(figsize=figsize)
        plot_tsne(
            temp_embeddings1,
            temp_numeric_labels1,
            perplexity=perplexity,
            title=" ".join(title.split("_")),
            alpha=.7,
            rev_map=rev_map1,
            cmap='magma',
            legend=False,
        )
        plt.tight_layout()
        plt.savefig(
            f'{save_path}/{save_name}_t_sne_{perplexity}.pdf'
        )
    visualize_pca(temp_embeddings1, temp_numeric_labels1)
    plt.tight_layout()
    plt.savefig(
        f'{save_path}/{save_name}_PCA.pdf'
    )

def visualize_latent_by_gene():
    N_GENES=50
    N_TRANSCRIPTS=5
    
    map = load_transcript_map(mouse=True)
    # app_h = load_appris(unique_transcripts=False)

    # create gene_root_go_prior
    dfs = []
    # get top count of genes
    dir = '/h/phil/Documents/01_projects/contrastive_rna_representation/annotation_data/'
    refseq_location_human=f"{dir}/refseq_gencode_files/human_comprehensive_gencode_v41_hg38.tsv"
    chromosomes_to_use = ["chr{}".format(i) for i in range(23)]
    df = RefseqDataset.load_refseq_as_df(refseq_location_human, chromosomes_to_use=chromosomes_to_use, mini=False)
    df['gene_name'] = df['name2']
    df['Transcript ID'] = [x.split('.')[0] for x in df['name']]
    
    genes = df['gene_name'].value_counts()[:N_GENES].keys()
    df = df[df['gene_name'].isin(genes)]
    df['species'] = 'human'

    dir = '/h/phil/Documents/01_projects/contrastive_rna_representation/annotation_data/'
    refseq_location_mouse=f"{dir}/refseq_gencode_files/mouse_comprehensive_gencodevm25_mm10.tsv"
    chromosomes_to_use = ["chr{}".format(i) for i in range(23)]
    df2 = RefseqDataset.load_refseq_as_df(refseq_location_mouse, chromosomes_to_use=chromosomes_to_use, mini=False)
    df2['gene_name'] = df2['name2']
    df2['Transcript ID'] = [x.split('.')[0] for x in df2['name']]
    
    genes = df2['gene_name'].value_counts()[:N_GENES].keys()
    df2 = df2[df2['gene_name'].isin(genes)]
    df2['species'] = 'mouse'
    df = pd.concat([df, df2])
  
    for i, (name, group) in enumerate(df.groupby('gene_name')):
        print(name)
        group = group.sample(frac=1).reset_index(drop=True)
        group = group.iloc[:N_TRANSCRIPTS]
        print(name, i)
        dfs.append(group)

    sampled_by_gene = pd.concat(dfs)
    print(len(sampled_by_gene))
    sampled_by_gene['term'] = sampled_by_gene['species']

    contrastive_epochs = list(range(4, 500, 40))

    for epoch in contrastive_epochs:
        model_path = (
            "rna_contrast_5-9.1_adamw_wd1e-6-pool_avgpool-DCLdev_4"
            "-d_0.1-seed_0-bs_768-less_dilated_small2-lr_0.005-e_500"
        )
        model, model_name = load_model(epoch=epoch, model_path=model_path)

        create_viz_for_model_by_gene(
            model,
            sampled_by_gene,
            map,
            title=f'Epoch {epoch}',
            directory_name=f'n_genes_{N_GENES}_plots_epochs_{contrastive_epochs[0]}:{contrastive_epochs[-1]}',
        )

def load_model(
    epoch=532, 
    model_path = (
        "rna_contrast_5-9.4_graft_wd3e-5-pool_avgpool-DCLdev_4"
        "-d_0.1-seed_0-bs_1536-less_dilated_small2-lr_0.01-e_600"
    )
    ):
    print(f"EPOCH: {epoch} {make_timestamp()}")
    # model_path = (
    #     "rna_contrast_5-5.0_11g_b2048_h512_wd1e-5_bn_mm-pool_avgpool"
    #     "-DCLdev_4-d_0.1-seed_0-bs_512-dilated_medium-lr_0.001-e_1000"
    # )
    model_name = f'9.4_graft_wd3e-5_less_dilated_small2_{epoch}'

    model = load_contrastive_model(
        model_path=model_path,
        contrastive_checkpoint=epoch,
    )

    return model, model_name

def visualize_model_latent_space():
    godf_mf, godf_cc, godf_bp = load_go_terms()
    app_h = load_appris()
    map = load_transcript_map()

    go_terms = [
        'mf',
        'cc', 'bp'
    ]
    go_names = [
        'Molecular Function',
        'Cellular Components', 'Biological Process'
    ]
    go_dfs = [
        godf_mf,
        godf_cc, godf_bp
    ]
    
    contrastive_epochs = list(range(4, 500, 40))

    for epoch in contrastive_epochs:
        model_path = (
            "rna_contrast_5-9.1_adamw_wd1e-6-pool_avgpool-DCLdev_4"
            "-d_0.1-seed_0-bs_768-less_dilated_small2-lr_0.005-e_500"
        )
        model, model_name = load_model(epoch=epoch, model_path=model_path)
    
        for go_tree, go_title, godf in zip(go_terms, go_names, go_dfs):

            one_below_root, two_below_root, three_below_root = load_go_root_terms(
                go_tree
            )
            create_viz_for_model_and_go_tree(
                model,
                godf,
                app_h,
                map,
                three_below_root,
                # title=model_name + "_" + go_title,
                title=go_title,
                n_go_terms=3,
                n_down_frm_root=3,
                directory_name=model_name,
                number_of_sample_positions=250
            )


def main():
    # visualize_model_latent_space()
    visualize_latent_by_gene()

if __name__ == "__main__":
    main()
