from typing import Optional
from math import sqrt
import torch
from torch import Tensor
from tqdm import tqdm
from inspect import signature
import pandas as pd
import numpy as np

from torch_geometric.nn import GNNExplainer, MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils import get_node_name

EPS = 1e-15

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

class GATExplainer(GNNExplainer):
    def __init__(self, model, epochs: int = 500, lr: float = 0.01, 
        num_hops: Optional[int] = 2, return_type: str = 'log_prob',
        feat_mask_type: str = 'feature', allow_edge_mask: bool = True, 
        log: bool = True, **kwargs):
        super(GATExplainer, self).__init__(model, epochs=epochs, lr=lr, num_hops=num_hops, 
                                    return_type=return_type, feat_mask_type=feat_mask_type, 
                                    allow_edge_mask=allow_edge_mask, log=log, **kwargs)
        self.devices = model.devices_available

    def edge_mask_symmetrize(self, edge_index, N):
        if N == 1: return
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    if module.__explain__:
                        mask_adj = torch.sparse_coo_tensor(edge_index, module.__edge_mask__, size=(N, N)).to_dense()
                        mask_adj = (mask_adj + mask_adj.T) / 2
                        module.__edge_mask__.data = mask_adj.to_sparse().values()

    def __set_masks__(self, x, edge_index):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)
        
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        edge_mask = torch.randn(E) * std
        if E != 0:
            mask_adj = torch.sparse_coo_tensor(edge_index, edge_mask.to(self.devices), size=(N, N)).to_dense()
            mask_adj = (mask_adj + mask_adj.T) / 2
            edge_mask = mask_adj.to_sparse().values()
        # self.loop_mask = edge_index[0] != edge_index[1]

        self.edge_mask = torch.nn.Parameter(edge_mask)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
                # module.__loop_mask__ = self.loop_mask

    def __loss__(self, node_idx, log_logits, pred_label):
        if pred_label[node_idx] == 1:
            log_logits = log_logits.log()
        else:
            log_logits = (1 - log_logits).log()
        loss = -log_logits[node_idx]

        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def explain_node(self, node, data, repeat=5, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.model.eval()
        self.__clear_masks__()
        # 为了每一次的对称化的边顺序都保持一致，方便与edge_mask对应
        num_nodes = data[0].x.size(0)
        num_edges = data[0].edge_index.size(1)
        print(f'Total Edges: {num_edges}')
        if num_edges != 0:
            adj = torch.sparse_coo_tensor(data[0].edge_index, data[0].edge_attr, size=(data[0].num_nodes, data[0].num_nodes, 1)).to_dense().squeeze().to_sparse()
            data[0].edge_index = adj.indices()
            data[0].edge_attr = adj.values().unsqueeze(1)

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(data)
            out = torch.squeeze(out)
            pred_label = torch.zeros(num_nodes)
            pred_label[out>0.5] = 1 
            print(f'Pred label: {pred_label[0].item()}, True label: {data[0].y[0, 1].item()}')
            pred_y = np.zeros(16165)
            pred_y[data[0].pos.cpu()] = pred_label

        for i in range(repeat):
            inner_data = [data[0].clone()]
            self.__set_masks__(inner_data[0].x, inner_data[0].edge_index)
            self.to(inner_data[0].x.device)

            if self.allow_edge_mask and num_edges != 0:
                parameters = [self.node_feat_mask, self.edge_mask]
            else:
                parameters = [self.node_feat_mask]
            optimizer = torch.optim.Adam(parameters, lr=self.lr)

            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                pbar.set_description(f'Explain node {node}')

            x = inner_data[0].x.clone()
            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                inner_data[0].x = x * self.node_feat_mask.sigmoid()
                log_logits = self.model(inner_data)
                loss = self.__loss__(0, log_logits, pred_label)
                loss.backward()
                optimizer.step()
                self.edge_mask_symmetrize(inner_data[0].edge_index, num_nodes)

                if self.log:  # pragma: no cover
                    pbar.update(1)

            if self.log:  # pragma: no cover
                pbar.close()

            node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
            edge_mask = self.edge_mask.cpu().detach().sigmoid().numpy()
            if i == 0:
                avg_node_feat_mask = node_feat_mask
                avg_edge_mask = edge_mask
            else:
                avg_node_feat_mask += node_feat_mask
                avg_edge_mask += edge_mask

        avg_node_feat_mask /= repeat
        avg_edge_mask /= repeat

        if num_edges != 0:
            edge_index = data[0].pos[data[0].edge_index].cpu().detach().numpy()
            edge_list = pd.DataFrame([edge_index[0], edge_index[1]], index=["gene1", "gene2"]).transpose()
            edge_list['importance'] = avg_edge_mask.reshape(-1, 1)
            edge_list = edge_list.sort_values(by=['importance'], ascending=False)
            edge_list['index'] = np.arange(num_edges)
            edge_list = edge_list[edge_list['index'] % 2 == 0]
            edge_list = edge_list.drop(columns=['index'])

            edge_mask = edge_list['importance'].to_numpy().reshape(-1, 1)
            edge_mask = torch.from_numpy(edge_mask)
            edge_index = np.zeros((2, num_edges//2))
            edge_index[0] = edge_list['gene1'].to_numpy().reshape(1, -1)
            edge_index[1] = edge_list['gene2'].to_numpy().reshape(1, -1)
            edge_index = torch.from_numpy(edge_index).long()
        else:
            edge_mask, edge_index, edge_list = None, None, None

        self.__clear_masks__()

        return node_feat_mask, edge_mask, edge_index, edge_list, pred_y
    
    def visualize_subgraph(self, node_idx, edge_index, edge_mask, num_all_nodes, y, pred_y, topk=20,
                           threshold=None, edge_y=None, **kwargs):
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        assert edge_mask.size(0) == edge_index.size(1)

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if edge_mask.shape[0] == 0:
            print("No edges.")
            return None, None

        if topk is not None and edge_mask.shape[0] > topk:
            edge_mask = edge_mask[:topk]
            edge_index = edge_index[:, :topk]

        sub_node = torch.cat([edge_index[0], edge_index[1]]).unique()
        if self.__flow__() == 'target_to_source':
            row, col = edge_index
        else:
            col, row = edge_index
        
        # 在子图中建立新的index
        node_idx = row.new_full((num_all_nodes, ), -1)
        node_idx[sub_node] = torch.arange(sub_node.size(0), device=row.device)
        edge_index = node_idx[edge_index]

        y = y[sub_node, :].to(torch.float) / y.max().item()
        pred_y = pred_y[sub_node]
        labels = np.array([])
        for i in range(y.shape[0]):
            if (y[i, 0] + y[i, 1]) == 0:
                labels = np.append(labels, 0)
            elif y[i, 0] == 1:
                labels = np.append(labels, -1)
            else:
                labels = np.append(labels, 1)

        if edge_y is None:
            edge_color = []
            cmap = mpl.cm.get_cmap('viridis_r')
            norm = mpl.colors.Normalize(vmin=torch.min(edge_mask), vmax=torch.max(edge_mask))
            for att in edge_mask.cpu().detach().numpy():
                edge_color.append(cmap(norm(att)))

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=labels, num_nodes=len(labels)).to('cpu')
        Graph = to_networkx(data, node_attrs=['y'],
                        edge_attrs=['att', 'edge_color'])
        Graph = Graph.to_undirected()
        sub_node = get_node_name(sub_node.tolist())
        mapping = {k: i for k, i in enumerate(sub_node)}
        largest_cc = max(nx.connected_components(Graph), key=len)
        labels = labels[list(largest_cc)]
        pred_y = pred_y[list(largest_cc)]
        Graph = nx.relabel_nodes(Graph, mapping)
        G = Graph.subgraph(max(nx.connected_components(Graph), key=len))

        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs['node_size'] = kwargs.get('node_size') or 40
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs['font_size'] = kwargs.get('font_size') or 10
        label_kwargs['verticalalignment'] = 'bottom'

        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="<->",
                    alpha=1,
                    color=data['edge_color'],
                    linewidth=2,
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3, rad=0.1",
                ))

        node_color = np.array([])
        for i in range(len(labels)):
            if labels[i] == -1:
                node_color = np.append(node_color, 'grey')
            elif labels[i] == 1:
                node_color = np.append(node_color, 'orange')
            elif labels[i] == 0 and pred_y[i] == 1:
                node_color = np.append(node_color, 'gold')
            elif labels[i] == 0 and pred_y[i] == 0:
                node_color = np.append(node_color, 'midnightblue')
        nx.draw_networkx_nodes(G, pos, node_color=node_color,
                                **node_kwargs)

        nx.draw_networkx_labels(G, pos, font_color='dimgrey', **label_kwargs)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        return ax, G
        