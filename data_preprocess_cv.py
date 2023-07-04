import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import torch as t
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from umap import UMAP
import config_load

from utils import minmax, get_node_idx, get_all_nodes

EPS = 1e-8

def arg_parse():
    parser = argparse.ArgumentParser(description="Data Preprocess.")
    parser.add_argument('-p', '--patient', dest='patient', help="to package patient data", default=False, nargs='*')
    parser.add_argument('-r', '--reverse', dest='reverse', help="to package reverse patient data", action="store_true")
    return parser.parse_args()

def get_cell_line(data_dir):
    if "Breast_Cancer" in data_dir:
        cell_line = "/MCF7"
    elif "Leukemia" in data_dir:
        cell_line = "/K562"
    elif "AML" in data_dir:
        cell_line = '/' + data_dir.split('/')[-1]
    elif "Pan" in data_dir:
        cell_line = "/Pan"
    else:
        print(f"Invalid directory {data_dir}.")
    return cell_line


def read_table_to_np(table_file, sep='\t', dtype=float, start_col=1):
    data = pd.read_csv(table_file, sep=sep)
    data = data.iloc[:, start_col:].to_numpy().astype(dtype)
    return data


def get_hic_mat(data_dir='data/Breast_Cancer_Matrix', drop_rate=0.0, reduce='svd', reduce_dim=5, resolution='10KB', type='ice', norm='log'):
    """
    Read Hi-C matrix from a csv file and do dimensionality reduction or normalization. Corresponding matrix should be put into certain directory first.

    Parameters:
    ----------
    data_dir:   str, default='data/Breast_Cancer_Matrix'
    drop_rate:  float, default=0.0. 
                The proportion of entries to drop randomly for robustness study, set from 0.0 to 0.9. 
    reduce:     str, {'svd', 'svdr', 'nmf', 't-sne', 'umap', 'isomap', 'lle', False}, default='svd'. 
                Method for dimensionality reduction. Use False for no reduction.
    reduce_dim: int, default=5. 
                Dimensionality after reduction.
    resolution: str, {'5KB', '10KB', '25KB'}, default='10KB'.
                The resolution of Hi-C data.
    type:       str, {'ice', 'int'}, default='ice'.
                The type of Hi-C data.
    norm:       {'log', 'square', 'binary'}
                Whether to do min-max normalization for reduced Hi-C data.

    Returns:
    numpy.ndarray
        A matrix of shape (num_nodes, reduce_dim) if `reduce` is not False, or (num_nodes, num_nodes) otherwise.
    """
    DEFAULT_RESO = '10KB'
    cell_line = get_cell_line(data_dir)
    sample_rate = str(round(10 * (1-drop_rate)))

    def get_hic_dir():
        if type == 'ice':
            if sample_rate == '10' and resolution == DEFAULT_RESO:
                hic_dir = data_dir + cell_line + "_Adjacent_Matrix_Ice"
            else:
                hic_dir = data_dir + "/drop_hic_ice/" + resolution + '/' + \
                    sample_rate + '_' + cell_line[1:] + "_ICE_downsample.csv"
        elif type == 'int':
            hic_dir = data_dir + cell_line + "_Adjacent_Matrix"
        print(f"Loading Hi-C matrix from {hic_dir} ......")

        return hic_dir

    def normalize_hic_data(data, method, reduce):
        """
        Normalize the input Hi-C matrix.

        Parameters:
        ----------
        data: numpy.ndarray
              The input Hi-C matrix to be normalized.
        method:  str, {'log', 'square', 'binary'}
              The normalization method to use.

        Returns:
        -------
        numpy.ndarray
            The normalized Hi-C matrix.
        """
        UP_THRESHOLD = 10000
        DOWN_THRESHOLD = 5
        LOG_THRESHOLD = 0.7
        if method == 'log':
            data = np.log10(data + EPS)
            if reduce != False: return data
            else: 
                data[data<LOG_THRESHOLD] = 0
        elif method == 'square':
            # Clip the data to the range of [1, 10000] and apply a power-law transformation
            data[data > UP_THRESHOLD] = 10000
            data[data <= DOWN_THRESHOLD] = 0
            data = data ** 0.1
        elif method == 'binary':
            data = np.log10(data + EPS)
            data[data < LOG_THRESHOLD] = 0
            data[data > LOG_THRESHOLD] = 1
        else:
            raise ValueError(f"Invalid use value: {method}")

        return data

    if reduce == 'n2v':
        # Read pre-trained N2V embedding from a file
        hic_data = read_table_to_np(f'/N2V_embedding_{reduce_dim}.csv')
        return minmax(hic_data)

    reducer_dict = {
        'svd': TruncatedSVD(n_components=reduce_dim, algorithm="arpack"),
        'svdr': TruncatedSVD(n_components=reduce_dim),
        'nmf': NMF(n_components=reduce_dim, init='nndsvd', solver='mu',
                   beta_loss='frobenius', max_iter=10000, tol=1e-6, l1_ratio=1),
        't-sne': TSNE(n_components=reduce_dim,
                      learning_rate='auto', init='pca'),
        'umap': UMAP(n_components=reduce_dim),
        'isomap': Isomap(n_components=reduce_dim),
        'lle': LocallyLinearEmbedding(n_components=reduce_dim),
    }

    hic_data = read_table_to_np(
        get_hic_dir(), sep='\t', dtype=float, start_col=1)
    hic_data = normalize_hic_data(hic_data, method=norm, reduce=reduce)
    hic_data += 8 if reduce == 'nmf' else 0 

    reducer = reducer_dict[reduce] if reduce else None
    hic_data = reducer.fit_transform(hic_data) if reduce else hic_data

    return minmax(hic_data, axis= 1 if reduce else -1)


def get_ppi_mat(ppi_name='CPDB', drop_rate=0.0, from_list=False, random_seed=42, reduce=False, pan=False):
    """
    Read PPI data from a csv file and construct PPI adjacent matrix.

    Parameters:
    ----------
    ppi_name:   str, {'CPDB', 'IRef', 'Multinet', 'PCNet', 'STRING'}, default='CPDB'. 
                Name of PPI network dataset. Corresponding matrix should be put into certain directory first.
    drop_rate:  float, default=0.0. 
                Drop rate for robustness study.
    from_list:  bool, default=False.
                Whether the PPI data is loaded from a preprocessed adjacency list instead of a matrix.
    random_seed:int, default=42.
    reduce:     bool, default=False.
                Whether to load ppi embedding got by Node2Vec.
    pan:        bool, default=False.
                Whether to use pan data in EMOGI.

    Returns:
    numpy.ndarray
        A ndarray(num_nodes, num_nodes) contains PPI adjacent matrix.
    """
    prefix = "data" if not pan else "pan_data"
    if reduce:
        ppi_dir = prefix + f"/{ppi_name}/N2V_ppi_embedding_15.csv"
        print(f"Loading PPI feature from {ppi_dir} ......")
        return read_table_to_np(ppi_dir)
    # Load PPI data from an edge list
    if from_list:
        ppi_dir = prefix + f"/{ppi_name}/{ppi_name}_edgelist.csv"
        print(f"Loading PPI matrix from {ppi_dir} ......")
        data = pd.read_csv(
            prefix + f"/{ppi_name}/{ppi_name}_edgelist.csv", sep='\t')
        # Load the gene names
        gene_list, gene_set = get_all_nodes(pan=pan)

        # Extract the edges that are also in the list of gene names
        if not pan:
            adj = [(row[1], row[2], row[3]) for row in data.itertuples()
                if row[1] in gene_set and row[2] in gene_set]
            conf = [row[4] for row in data.itertuples() if row[1]
                    in gene_set and row[2] in gene_set]
            if drop_rate:
                # Drop samples with stratification by confidence score
                adj, drop_adj = train_test_split(
                    adj, test_size=drop_rate, random_state=random_seed, stratify=conf)
            # Construct the adjacency matrix from the edges
            adj_matrix = pd.DataFrame(0, index=gene_list, columns=gene_list)
            for line in adj:
                adj_matrix.loc[line[0], line[1]] = line[2]
                adj_matrix.loc[line[1], line[0]] = line[2]
        else:
            adj = [(row[1], row[2]) for row in data.itertuples()
                if row[1] in gene_set and row[2] in gene_set]
            adj_matrix = pd.DataFrame(0, index=gene_list, columns=gene_list)
            for line in adj:
                adj_matrix.loc[line[0], line[1]] = 1
                adj_matrix.loc[line[1], line[0]] = 1
        adj_matrix.to_csv(prefix + f'/{ppi_name}/{ppi_name}_matrix.csv', sep='\t')
        data = adj_matrix.to_numpy()

        return data

    # Load PPI data from a matrix
    ppi_dir = prefix + f"/{ppi_name}/{ppi_name}_matrix.csv"
    print(f"Loading PPI matrix from {ppi_dir} ......")
    data = pd.read_csv(ppi_dir, sep="\t").to_numpy()[:, 1:]

    return data


def get_label(data_dir='data/Breast_Cancer_Matrix', reverse=False):
    """
    Read label data, where some nodes have labels and others do not. For the nodes with labels, change the labels from -1 to 0.

    Returns:
    labels:         ndarray(num_nodes, 2). First col is 1 for negative nodes, and second col is 1 for positive nodes.
    labeled_idx:    list(num_nodes). Indices of labeled nodes.
    """
    cell_line = get_cell_line(data_dir)
    label_dir = data_dir + cell_line
    label_dir += "-Label.txt" if not reverse else "-test-Label.txt"
    data = read_table_to_np(label_dir, dtype=int).transpose()[0]
    labeled_idx = []
    labels = np.zeros((len(data), 2), dtype=float)
    for i in range(len(data)):
        if data[i] != 0:
            labeled_idx.append(i)
            if data[i] == -1:
                labels[i][0] = 1
            else:
                labels[i][1] = 1
    return labels, labeled_idx


def get_node_feat(hic_feat=None, data_dir='data/Breast_Cancer_Matrix'):
    """
    Merge marker node features from a csv file with Hi-C features from get_hic_mat().

    Parameters:
    ----------
    hic_feat:   ndarray(num_nodes, n), default=None.
                Hi-C features from get_hic_mat. None to ignore Hi-C features.

    Returns:
    data:       Ndarray(num_nodes, num_features). Node features.
    pos:        Ndarray(num_nodes,). Node indices.
    """
    cell_line = get_cell_line(data_dir)
    feat = read_table_to_np(data_dir + cell_line +
                            "-Normalized-Nodefeature-Matrix.csv", sep=',')
    feat = feat[:, :10] if cell_line != "/Pan" else feat
    feat = np.concatenate((feat, hic_feat), axis=1) if hic_feat is not None else feat
    print(f"Feature matrix shape: {feat.shape}")
    pos = np.arange(feat.shape[0])
    return feat, pos


def construct_edge(mat):
    """
    Construct edges from adjacent matrix.

    Parameters:
    ----------
    mat:    ndarray(num_nodes, num_nodes).
                PPI matrix from get_ppi_mat().

    Returns:
    edges:      list(num_edges, 2). 
    edge_dim:   int.
                Dim of edge features.
    val:        list(num_edges, ).
                Edge features(=[1] * num_edges in current version).
    """
    num_nodes = mat.shape[0]
    edges = []
    val = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if mat[i, j] > 0:
                edges.append([i, j])
                val.append(mat[i, j])

    edge_dim = 1
    edges = np.transpose(edges)
    val = np.reshape(val, (-1, edge_dim))

    return edges, edge_dim, val


def build_pyg_data(node_mat, node_lab, mat, pos):
    x = t.tensor(node_mat, dtype=torch.float)
    y = t.tensor(node_lab, dtype=torch.long)
    pos = t.tensor(pos, dtype=torch.int)
    edge_index, edge_dim, edge_feat = construct_edge(mat)
    edge_index = t.tensor(edge_index, dtype=torch.long)
    edge_feat = t.tensor(edge_feat, dtype=torch.float)
    data = Data(x=x.clone(), y=y.clone(), edge_index=edge_index,
                edge_attr=edge_feat, pos=pos, edge_dim=edge_dim)
    print(
        f"Number of edges: {data.num_edges}, Dimensionality of edge: {edge_dim},\nNubmer of nodes: {data.num_nodes}")

    return data


class CancerDataset(InMemoryDataset):
    def __init__(self, data=None):
        super(CancerDataset, self).__init__('.', None, None)
        self.data = data or get_data()
        self.data.num_classes = 2

        self.data, self.slices = self.collate([self.data])

    def get_idx_split(self, i):
        train_idx = torch.where(self.data.train_mask[:, i])[0]
        test_idx = torch.where(self.data.test_mask)[0]
        valid_idx = torch.where(self.data.valid_mask[:, i])[0]

        return {
            'train': train_idx,
            'test': test_idx,
            'valid': valid_idx
        }

    def get_feature_dict(self):
        feature_idx = {'ATAC': [0],
                       'CTCF': [1, 2, 3],
                       'H3K4me3': [4, 5],
                       'H3K27ac': [6, 7],
                       'SNV': [8],
                       'CNV': [9],
                       'Hi-C': [i for i in range(10, self.data.num_node_features)]}
        for i in range(len(feature_idx['Hi-C'])):
            feature_idx[f'Hi-C-{i+1}'] = [i + feature_idx['Hi-C'][0]]
        for i in range(len(feature_idx['CTCF'])):
            feature_idx[f'CTCF-{i+1}'] = [i + feature_idx['CTCF'][0]]
        for i in range(len(feature_idx['H3K4me3'])):
            feature_idx[f'H3K4me3-{i+1}'] = [i + feature_idx['H3K4me3'][0]]
        for i in range(len(feature_idx['H3K27ac'])):
            feature_idx[f'H3K27ac-{i+1}'] = [i + feature_idx['H3K27ac'][0]]

        return feature_idx
    

    def feature_disturb(self, disturb_dict, random_seed=42):  # disturb_dict = {"add": ['PPI'], "remove": ['Hi-C'], "random": ['ATAC']}
        print(f"Doing below distrubs: \n {disturb_dict}")
        indices = set(torch.arange(self.data.num_node_features).tolist())
        feature_dict = self.get_feature_dict()
        remove_idx, random_idx = [], []
        for feature in disturb_dict["random"]:
            random_idx.extend(feature_dict[feature])
        for feature in disturb_dict["remove"]:
            remove_idx.extend(feature_dict[feature])
        if random_idx:
            random_idx = torch.tensor(random_idx, dtype=torch.long)
            np.random.seed(random_seed)
            random_feat = minmax(np.random.normal(0, 0.1, size=(self.data.x.shape[0], random_idx.shape[0])))
            self.data.x[:, random_idx] = torch.tensor(random_feat, dtype=torch.float)
        reserved_idx = torch.tensor(list(indices - set(remove_idx)), dtype=torch.long)
        self.data.x = self.data.x[:, reserved_idx]

        if 'PPI' in disturb_dict['add']:
            ppi_feat = get_ppi_mat(reduce=True)
            self.data.x = torch.cat([self.data.x, torch.tensor(ppi_feat, dtype=torch.float)], dim=1)

    def network_disturb(self, drop_rate=0.0, random_seed=42):
        num_edges = self.data.edge_attr.shape[0]
        np.random.seed(random_seed)
        sample_idx = np.random.sample(np.arange(num_edges), num_edges*(1-drop_rate))
        self.data.edge_index, self.data.edge_attr = self.data.edge_index[:, sample_idx], self.data.edge_attr[sample_idx]


    def update_hic(self, configs):
        hic_feat = get_hic_mat(data_dir=configs["data_dir"], drop_rate=configs["hic_drop_rate"], reduce=configs["hic_reduce"], 
                               reduce_dim=configs["hic_reduce_dim"], resolution=configs["resolution"], type=configs["hic_type"], norm=configs["hic_norm"])
        feature_dict = self.get_feature_dict()
        self.data.x = self.data.x[:, :feature_dict["Hi-C"][0]]
        self.data.x = torch.cat([self.data.x, torch.tensor(hic_feat, dtype=torch.float)], dim=1)

    def perturb_features(self, nodes, perturbs):
        indices_x = []
        indices_y = []
        values = []
        node_idxs = get_node_idx(nodes)
        feature_idx = self.get_feature_dict()
        for node_idx in node_idxs:
            for feature, value in perturbs.items():
                indices_x.append(node_idx)
                indices_y.append(feature_idx[feature][0])
                values.append(value)
        indices = (torch.LongTensor(indices_x), torch.LongTensor(indices_y))
        values = torch.Tensor(values)
        self.data.x[indices] = values

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def create_cv_dataset(train_idx_list, valid_idx_list, test_idx, hic_data=None, ppi_data=None):
    data = [hic_data, ppi_data] if hic_data and ppi_data else hic_data or ppi_data
    num_nodes = ppi_data.num_nodes if ppi_data else hic_data.num_nodes
    num_folds = len(train_idx_list)

    train_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    valid_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for i in range(num_folds):
        train_mask[train_idx_list[i], i] = True
        valid_mask[valid_idx_list[i], i] = True
    test_mask[test_idx] = True

    if isinstance(data, list): # data[0]和data[1]共享mask
        data[0].train_mask = data[1].train_mask = torch.tensor(
            train_mask, dtype=torch.bool)
        data[0].valid_mask = data[1].valid_mask = torch.tensor(
            valid_mask, dtype=torch.bool)
        data[0].test_mask = data[1].test_mask = torch.tensor(
            test_mask, dtype=torch.bool)
        data[0].unlabeled_mask = data[1].unlabeled_mask = ~torch.logical_or(
            data[0].train_mask[:, 0], torch.logical_or(data[0].valid_mask[:, 0], data[0].test_mask))
        cv_dataset = dict(hic=CancerDataset(
            data=data[0]), ppi=CancerDataset(data=data[1]))
    else:
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data.unlabeled_mask = ~torch.logical_or(
            data.train_mask[:, 0], torch.logical_or(data.valid_mask[:, 0], data.test_mask))
        cv_dataset = CancerDataset(data=data)

    return cv_dataset

def post_process(dataset, configs, disturb_list=None):

    def hic_needed_update(configs):
        needed = (configs["hic_drop_rate"] != 0.0 or 
                  configs["hic_reduce"] != 'svd' or 
                  configs["hic_reduce_dim"] != 5 or 
                  configs["resolution"] != '10KB' or 
                  configs["hic_type"] != 'ice')
        return needed and (configs["hic_reduce"] != False)
    
    random_seed = configs["random_seed"]

    if isinstance(dataset, CancerDataset):
        if configs["ppi_drop_rate"]:
            print("Dropping PPI edges ......")
            dataset.network_disturb(configs["ppi_drop_rate"], random_seed)
        if hic_needed_update(configs):
            print("Updating Hi-C for experimental need ......")
            dataset.update_hic(configs)
        if disturb_list is not None:
            dataset.feature_disturb(disturb_list, random_seed)


def get_data(configs, disturb_list=None, stable=True):
    cv_folds = configs["cv_folds"]
    data_dir = configs["data_dir"]  # directory of data
    hic = configs["hic"]
    hic_drop_rate = configs["hic_drop_rate"]
    hic_norm = configs["hic_norm"]
    hic_reduce = configs["hic_reduce"] if configs["graph"] not in ["dual", "plusc", "onlyc"] else False
    hic_reduce_dim = configs["hic_reduce_dim"]
    hic_type = configs["hic_type"]
    load_data = configs["load_data"]
    ppi = configs["ppi"]
    ppi_drop_rate = configs["ppi_drop_rate"]
    random_seed = configs["random_seed"]
    resolution = configs['resolution']
    if disturb_list is None:
        disturb_list = {"add"    : [],
                        "remove" : [],
                        "random" : [],}
    else:
        for key in ['add', 'remove', 'random']:
            if key not in disturb_list: disturb_list[key] = []

    cell_line = get_cell_line(data_dir)
    pan = cell_line == "/Pan"

    def get_dataset_dir(stable):
        if hic_reduce:
            if configs['reverse']:
                dataset_suffix = f'_{ppi}_dataset_r_final' if stable else f'_{ppi}_dataset_r'
            else:
                dataset_suffix = f"_{ppi}_dataset_final" if stable else f"_{ppi}_dataset"
        else:
            dataset_suffix = "_dataset_" + configs["graph"] # "dual", "onlyc", "plusc"

        dataset_dir = os.path.join(
            data_dir, cell_line[1:] + dataset_suffix + '.pkl')
        
        return dataset_dir

    if load_data:
        dataset_dir = get_dataset_dir(stable)
        print(f"Loading dataset from: {dataset_dir} ......")
        with open(dataset_dir, 'rb') as f:
            cv_dataset = pickle.load(f)
        if not hic:
            disturb_list["remove"].append("Hi-C")
            
        post_process(cv_dataset, configs, disturb_list)

        return cv_dataset

    hic_mat = get_hic_mat(data_dir=data_dir, drop_rate=hic_drop_rate, reduce=hic_reduce,
                            reduce_dim=hic_reduce_dim, resolution=resolution, type=hic_type, norm=hic_norm) if hic else None

    ppi_mat = get_ppi_mat(
        ppi, drop_rate=ppi_drop_rate, from_list=False, random_seed=random_seed, pan=pan) if ppi else None

    node_mat, pos = get_node_feat(hic_feat=hic_mat if hic_reduce else None, data_dir=data_dir)

    node_lab, labeled_idx = get_label(data_dir, reverse=configs['reverse'])
    labeled_lab = [node_lab[i][1] for i in labeled_idx]

    train_idx_list, valid_idx_list = [], []
    train_valid_idx, test_idx, train_valid_lab, test_lab = train_test_split(
        labeled_idx, labeled_lab, test_size=0.25, stratify=labeled_lab, random_state=random_seed)
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_seed)

    for train_labeled_idx, valid_labeled_idx in skf.split(train_valid_idx, train_valid_lab):
        valid_idx_list.append([train_valid_idx[i] for i in valid_labeled_idx])
        train_idx_list.append([train_valid_idx[i] for i in train_labeled_idx])
    if configs["graph"] != "plusc":
        hic_data = build_pyg_data(
            node_mat, node_lab, hic_mat, pos) if configs["graph"]=="dual" or configs["graph"]=="onlyc" else None
        ppi_data = build_pyg_data(node_mat, node_lab, ppi_mat, pos) if ppi else None
    else:
        mat = minmax(hic_mat + ppi_mat, -1)
        hic_data = None
        ppi_data = build_pyg_data(node_mat, node_lab, mat, pos)
    cv_dataset = create_cv_dataset(
        train_idx_list.copy(), valid_idx_list.copy(), test_idx.copy(), hic_data, ppi_data)

    dataset_dir = get_dataset_dir(stable=False)
    print(f'Finished! Saving dataset to {dataset_dir} ......')
    with open(dataset_dir, 'wb') as f:
        pickle.dump(cv_dataset, f)

    post_process(cv_dataset, configs, disturb_list)

    return cv_dataset


if __name__ == "__main__":
    configs = config_load.get()
    args = arg_parse()
    configs["load_data"] = False
    if args.reverse:
        configs["reverse"] = True
    if args.patient:
        for patient in args.patient:
            configs["data_dir"] = f'data/AML_Matrix/{patient}'
            data = get_data(configs)
        sys.exit()
    data = get_data(configs)