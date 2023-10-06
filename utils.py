import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def minmax(input, axis=1):
    """
    Do minmax normalization for input 2d-ndarray.

    Parameters:
    ----------
    input:       np.ndarray.
                The input 2d-ndarray.
    axis:       int, default=1.
                The axis should be normalized. Default is 1, that is do normalization along with the column.

    Returns:
    A ndarray after minmax normalization.
    """
    scaler = MinMaxScaler()
    if axis == 1:
        output = scaler.fit_transform(input)
    elif axis == 0:
        output = scaler.fit_transform(input.T).T
    elif axis == -1:
        output = (input - np.min(input)) / (np.max(input) - np.min(input))

    return output


def sum_norm(input, axis=1):
    """
    Do normalization for an input 2d-ndarray, making the sum of every row or column equals 1.

    Parameters:
    ----------
    input:       ndarray.
                The input 2d-ndarray.
    axis:       int, default=1.
                The axis should be normalized. Default is 1, that is do normalization along with the column.

    Returns:
    A ndarray after normalization.
    """
    axis_sum = input.sum(axis=1-axis, keepdims=True)
    return input / axis_sum

def get_all_nodes(pan=False):
    dir = "data/Gene-Name.txt" if not pan else f"data/{pan}/Gene-Name.txt"
    with open(dir) as f:
        txt = f.readlines()
    gene_list = [line.strip() for line in txt[1:]]  # skip the first line
    gene_set = set(gene_list)

    return gene_list, gene_set


def get_labeled_nodes(cancer, labels=[-1, 1], return_labels=False, patient=None):
    """
    Get a list of labeled nodes for a given cancer type.

    Parameters:
    ----------
    cancer:         str.
                    The name of the cancer dataset. Available options are "Breast_Cancer" and "Leukemia".
    labels:         list, default=[-1, 1].
                    The labels to include in the output list. Available options are -1 (negative label) and 1 (positive label).
    return_labels:  bool, default=False.
                    Whether to return the labels of the nodes as well.

    Returns:
    A node list contains labeled nodes' name. If return_labels is True, the output is a tuple (node_list, label_list).
    """
    HIC_NAMES = {"Breast_Cancer": "MCF7", "Leukemia": "K562"}
    file_name = f"data/{cancer}_Matrix/{HIC_NAMES[cancer]}-Label.txt" if not patient else f"data/{cancer}_Matrix/{patient}/{patient}-Label.txt"
    data = pd.read_csv(file_name, sep='\t')
    data = data[data['label'] != 0]
    data = data[data['label'].isin(labels)]
    node_list = data['gene_name'].to_list()
    print(len(node_list))
    
    if not return_labels:
        return node_list
    label_list = data['label'].to_list()

    return node_list, label_list


def get_gene_list(rename=False):
    gene_list = pd.read_csv("data/Gene-Name.txt")
    return gene_list.rename(columns={'gene_name' : 'Gene Name'}) if rename else gene_list

def get_node_idx(node_list):
    """
    Get node indices from node name list.

    Parameters:
    ----------
    node_list:  list.
                Node name list.

    Returns:
    Node indices list.
    """
    dir = "data/Gene-Name.txt"
    gene_list = pd.read_csv(dir)
    gene_list.set_index('gene_name', inplace=True)
    gene_index = np.arange(gene_list.shape[0])
    gene_list['gene_index'] = gene_index
    node_idx_list = []
    for node in node_list:
        if node not in gene_list.index:
            print(f"{node} is not in gene list!")
            continue
        node_idx = gene_list.loc[node, 'gene_index']
        node_idx_list.append(node_idx)

    return node_idx_list

def get_node_name(node_idx_list):
    """
    Get node name from node indices list.

    Parameters:
    ----------
    node_idx_list:  list.
                    Node indices list.

    Returns:
    Node name list.
    """
    gene_list = pd.read_csv("data/Gene-Name.txt")
    assert all(x >= 0 and x < gene_list.shape[0] for x in node_idx_list), "Elements are out of bounds."

    node_name_list = gene_list.iloc[node_idx_list, 0].to_list()

    return node_name_list
