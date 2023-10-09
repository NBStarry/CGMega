# CGMega
CGMega is a graph attention-based deep learning framework for cancer gene module dissection. CGMega leverages a combination of multi-omics data cross genome, epigenome, proteome and especially three-dimension (3D) genome levels.

CGMega was written in Python 3.8, and should run on any OS that support pytorch and pyg. Training is faster on a GPU with at least 24G memory to reproduce our results.

## Documentation
CGMega documentation is available through [Documentation](https://sunyolo.github.io/CGMega.github.io/).

## Conda Environment
We recommend using conda to configure the code runtime environment, this could take 10-30 minutes:
```
conda create -n cgmega python=3.8.12
conda install pytorch==1.9.1 -c pytorch
pip install torch_geometric==2.0.3 transformers wandb jupyter
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.9.1+cu113.html
pip install torch_sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+cu113.html
```
Install commmands of torch_scatter and torch_sparse should be adjusted according to pytorch and cuda version, see [PyG 2.0.3 Installation](https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/installation.html)

## Installation
We recommend getting CGMega using Git from our Github repository through the following command:

```
git clone https://github.com/NBStarry/CGMega.git
```
>The software package of CGMega is still under testing. We are going to release a stable version in the near future.

To verify a successful installation, just run:
```
python main.py -cv -l  # Importing the relevant libraries may take a few minutes.
```

## Tutorial
This tutorial demonstrates how to use CGMega functions with a demo dataset (MCF7 cell line as an example). 
Once you are familiar with CGMega’s workflow, please replace the demo data with your own data to begin your analysis. 
[Tutorial notebook](https://github.com/NBStarry/CGMega/tree/main/Tutorial.ipynb) is available now.

### How to prepare input data

We recommend getting started with CGMega using the provided demo dataset. When you want to apply CGMega to your own multi-omics dataset, please refer to the following tutorials to learn how to prepare input data.

Overall, the input data consists of two parts: the graph, constructed from PPI and the node feature including condensed Hi-C features, SNVs, CNVs frequencies and epigenetic densities.

 If you are unfamiliar with CGMega, you may start with our data used in the paper to save your time. For MCF7 cell line, K562 cell line and AML patients, the input data as well as their label information are uploaded [here](https://github.com/NBStarry/CGMega/tree/main/data). If you start with any one from these data, you can skip the _step 1_ about _How to prepare input data_.
 The following steps from 1.1~1.3 can be found in our source code [data_preprocess_cv.py](https://github.com/NBStarry/CGMega/blob/main/data_preprocess_cv.py).

>The labels should be collected yourself if you choose analyze your own data.

#### Hi-C data embedding
Before SVD, the Hi-C data should go through: 
1. processing by [NeoLoopFinder](https://github.com/XiaoTaoWang/NeoLoopFinder) to remove the potential effects of structural variation; 
2. normalization using [ICE]( https://bitbucket.org/mirnylab/hiclib) correction to improve the data quality. 

If you are new to these two tools, please go through these document in advance.
[tutorial for NeoLoopFinder (need link)](./)
[tutorial for Hi-C normalization (need link)](./)

The parameters and main functions used in NeoLoopFinder are listed as below:

---

Parameters:

- input file format: .cool or .mcool
- resolution: 10Kb
- binsize: 10000
- ploidy: 2
- enzymes: MobI
- genome: hg38

Please choose parameters by [NeoLoopFinder](https://github.com/XiaoTaoWang/NeoLoopFinder) to suit your data. An example is available in [batch_neoloop.sh](https://github.com/NBStarry/CGMega/blob/main/data/AML_Matrix/batch_neoloop.sh)

---

Then we implement ICE correction following [Imakaev, Maxim et al.](https://www.nature.com/articles/nmeth.2148) and this step has beed packaged in one-line command as `content from xuxiang`.

After the corrections by NeoLoopFinder and ICE, we then condense the chromatin information in Hi-C data.
The defualt way for Hi-C dimention reduction is Singular Value Decomposition (SVD).
```
hic_mat = get_hic_mat(
            data_dir=data_dir,
            drop_rate=hic_drop_rate,   # default=0.0
            reduce=hic_reduce,         # default='svd'
            reduce_dim=hic_reduce_dim, # efault=5
            resolution=resolution,     # default='10Kb', depends on your own Hi-C data
            type=hic_type,             # default='ice'
            norm=hic_norm              # default='log'
          )
```

Now we get the reduced Hi-C data as below:

| gene_name | HiC-1       | HiC-2       | HiC-3       | HiC-4       | HiC-5       |
| --------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| OR4F5     | 1           | 0.000160124 | 0.224521596 | 0.240359624 | 0.48014354  |
| SAMD11    | 0.983033738 | 0.022808685 | 0.215826884 | 0.232252798 | 0.483052779 |
| NOC2L     | 0.982147132 | 0.037410285 | 0.187267998 | 0.224593303 | 0.464582689 |
| KLHL17    | 1           | 0.000160124 | 0.224521596 | 0.240359624 | 0.48014354  |
| PLEKHN1   | 1           | 0.000160124 | 0.224521596 | 0.240359624 | 0.48014354  |
| ISG15     | 0.990748546 | 0.013588959 | 0.217608283 | 0.228825281 | 0.481511407 |
| AGRN      | 0.974279543 | 0.055531025 | 0.190370936 | 0.218615449 | 0.485405181 |
| C1orf159  | 0.96152902  | 0.062132207 | 0.185860829 | 0.212024252 | 0.454002735 |
| TTLL10    | 0.991588209 | 0.010710517 | 0.215210022 | 0.23584837  | 0.479124017 |
| SDF4      | 0.974902568 | 0.039697561 | 0.176009104 | 0.226073947 | 0.49244549  |

#### Other omics data
code for other omics-data processing.

#### PPI graph construction

Then,we read the PPI data and transform it into a graph through the following commands.

 ```
ppi_mat = get_ppi_mat(ppi, drop_rate=ppi_drop_rate, from_list=False, random_seed=random_seed, pan=pan) if ppi else None
edge_index, edge_dim, edge_feat = construct_edge(ppi_mat)
 ```

The basic properties of this graph (on MCF7 ) will be:

- number of edges: 1,000,000 (or so?)
- number of nodes: 16,165
- feature of any node (e.g., BRCA1): 

| gene_name | ATAC   | CTCF-1 | CTCF-2 | CTCF-3 | H3K4me3-1 | H3K4me3-2 | H3K27ac-1 | H3K27ac-2 | Means-SNV | Means-CNV | Hi-C-1 | Hi-C-2 | Hi-C-3 | Hi-C-4 | Hi-C-5 |
| --------- | ------ | ------ | ------ | ------ | --------- | --------- | --------- | --------- | --------- | --------- | ------ | ------ | ------ | ------ | ------ |
| BRCA1     | 0.8721 | 0.4091 | 0.2511 | 0.3964 | 0.8850    | 0.9591    | 0.9387    | 0.8595    | 0.6185    | 0.2460    | 0.8972 | 0.1935 | 0.0946 | 0.1747 | 0.5598 |


Now the input data is prepared. Let'a go for the model training!

### Model training

This section demonstrates the GAT-based model architecture of CGMega and how to train CGMega.

---

### CGMega framework

![image](https://github.com/NBStarry/CGMega/blob/main/img/model_architecture.png)

---

### Hyperparameters

- To reduce the number of parameters and make training feasible within time and resource constraints, the input graphs were sampled using neighbor sampler. The subgraphs included all first and second order neighbors for each node and training was performed on these subgraphs.
- The learning rate is increased linearly from 0 to 0.005 for the first 20% of the total iterations.
- warm-up strategy for learning rate is employed during the initial training phase.
- To prevent overfitting and over smoothing, an early stop strategy is adopted. If the model's performance on the validation set dose not improve for a consecutive 100 epochs, the training process stops.
- Dropout is used and the dropout rate is set to 0.1 for the attention mechanism and 0.4 for the other modules.
- Max pooling step size is 2. After pooling, the representation had 192 dimensions.

---

### System and Computing resources

| Item       | Details          |
| ---------- | ---------------- |
| System     | Ubuntu 20.04.1 LTS |
| RAM Memory | 256G               |
| GPU Memory | NVIDIA GeForce RTX, 24G     |
| Time       | ~ 30m             |

```note
The above table reports our computing details during CGMega development and IS NOT our computing requirement.

If your computer does not satisfy the above, you may try to lower down the memory used during model training by reduce the sampling parameters, the batch size or so on. 

We are going to test CGMega under more scenarios like with different models of GPU or memory limits to update this table.

```
## Interpretation

After prediction, you can do analyses as following to interpret your results.

#### 1. identify the gene module

For each gene, GNNExplainer identified a subgraph G that is most influential to the preiction of its identity from both topological integration and multi-omic information.
This subgraph consists of two parts: 

- i) the core subgraph that consists of the most influential pairwise relationships for cancer gene prediction, and
- ii) the 15-dimension importance scores that quantify the contributions of each gene feature to cancer gene prediction.

```note
The above module identification is calculated at the level of individual genes.
High-order gene modules reported in our work are constructed with the pairwise-connected individual gene modules.
```

#### 2. calculate the Representative Feature

According to the feature importance scores calculated by GNNExplainer, we defined the representative features (RFs) for each gene as its features with relatively prominent importance scores.
In detail, for a given gene, among its features from ATAC, CTCF, H3K4me3 and H3K27ac, SNVs, CNVs and Hi-C, if a feature is assigned with importance score as 10 times higher than the minimum score,it will be referred to as the RF of this gene.
A graphic illustration is shown as below:

![image](https://github.com/NBStarry/CGMega/blob/main/img/RF_calculation.png)

#### 3. explore the relationships between different gene modules

CGMega serves as a tool to help researchers explore the relationships between individual modules of genes. Such kind of view of high-order gene modules may also helps to find something new.
This is also useful especially when we do not know how to integrate some isolated knowledges into a whole even they are already well-studied under different cases. 

For example, BRCA1 and BRCA2 these two genes act as different roles in common pathway of genome protection and this also exhibited on the topology of their gene modules.
In brief, BRCA1, as a pleiotropic DDR protein working in broad stages of DNA damage response (DDR), was also widely connected with another 20 genes. 
In contrast, BRCA2, as the mediator of the core mechanism of homologous recombination (HR), was connected with other genes via ROCK2, an important gene that directly mediates HR repair.
Moreover, SNV was the RF for both BRCA1 and BRCA2. We also observed a high-order gene module combined from BRCA1 gene module and BRCA2 gene module through three shared genes including TP53, SMAD3 and XPO1.

![image](https://github.com/NBStarry/CGMega/blob/main/img/example.png)

## Questions and Code Issues
If you are having problems with our work, please use the [Github issue page](https://github.com/NBStarry/CGMega/issues).