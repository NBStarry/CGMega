import os
import argparse
import time
import sys

import torch as t
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.loader import NeighborLoader

import config_load
from data_preprocess_cv import get_data, CancerDataset
from utils import get_node_name, get_node_idx, sum_norm, get_labeled_nodes
from model import CGMega
from explainer import GATExplainer


def arg_parse():
    parser = argparse.ArgumentParser(description="GNNExplainer arguments.")
    parser.add_argument('-s', "--stable-exp", dest='s', action='store_true')
    parser.add_argument('--heatmap', dest='heatmap', action='store_true')
    parser.add_argument('-pt', "--patient", dest='patient', help="Patient ID", default=None, nargs='*')
    parser.add_argument('-f', '--format', dest='format', default='png')
    parser.add_argument('-g', "--gpu", dest="gpu", default=3)
    parser.add_argument('-k', "--known", dest='known', action='store_true')
    parser.add_argument('-m', "--model", dest='model')
    parser.add_argument('-n', '--node_list', dest='node_list', default=None)
    parser.add_argument('-o', '--output', dest='output', default=None)
    return parser.parse_args()


def select_best_model(dir):
    ckpts = os.listdir(dir)
    auprcs = list(map(lambda x: float(x.split('_')[0]), ckpts))
    best_ckpt = f'{max(auprcs):.4f}'
    for ckpt in ckpts:
        if best_ckpt in ckpt: 
            print(f'Best model is {ckpt}')
            return os.path.join(dir, ckpt)
        

class Single_Node_Explain():
    def __init__(self, dataset, explainer, node, out_dir, label, viz_subgraph=True, repeat=1, repeat_times=5, format='png') -> None:
        self.dataset = dataset
        self.node = node
        self.repeat = repeat
        self.out_dir = out_dir
        self.explainer = explainer
        self.repeat_times = repeat_times
        self.viz_subgraph = viz_subgraph
        self.label = label
        self.format = format
        
        self.node_idx = self.if_have_explained
        self.out_name = f'{node}' if repeat == 1 else [f'{node}_{i}' for i in range(repeat)]

    @property
    def if_have_explained(self):
        node_idx = get_node_idx([self.node])
        if os.path.exists(os.path.join(self.out_dir, 'feature_mask')):
            explained_node_list = os.listdir(os.path.join(self.out_dir, 'feature_mask'))
            explained_node_list = list(map(lambda x: x[:-4], explained_node_list))
            if node_idx == []:
                print(f'Do not have node: {self.node}')

            if self.node in explained_node_list and self.repeat == 1:
                print(f"already have {self.node}")
                return []
            
        return node_idx
    
    @property
    def get_loader(self):
        data = self.dataset[0]
        loader = NeighborLoader(data, batch_size=1, num_neighbors=[-1, -1], input_nodes=t.tensor([self.node_idx], dtype=t.long), shuffle=False, directed=False)
        return [next(iter(loader))]
    
    @property
    def explain_node(self, ):
        return self.explainer.explain_node(self.node, self.get_loader, repeat=self.repeat_times)
    
    def save_feat_mask(self, feat_mask, out_name):
        feat_csv = pd.DataFrame(feat_mask, columns=self.label, index=[self.node])
        save_dir = os.path.join(self.out_dir, "feature_mask")
        if not os.path.exists(save_dir): 
            print(f'{save_dir} does not exists, creating {save_dir}')
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, f'{out_name}.csv')
        feat_csv.to_csv(save_dir, sep='\t')

    def generate_subgraph(self, edge_index, edge_mask, pred_y, out_name,):
        plt.clf()
        ex, G = self.explainer.visualize_subgraph(self.node_idx, edge_index=edge_index, edge_mask=edge_mask, num_all_nodes=self.dataset[0].num_nodes,
                y=self.dataset[0].y, pred_y=pred_y)
        plt.title(self.node)
        save_dir = os.path.join(self.out_dir, "subgraph")
        if not os.path.exists(save_dir): 
            print(f'{save_dir} does not exists, creating {save_dir}')
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{out_name}.' + self.format), transparent=True, format=self.format)
    
    def save_edge_mask(self, edge_index, edge_list, out_name):
        node1 = get_node_name(edge_index[0].cpu().detach().numpy())
        node2 = get_node_name(edge_index[1].cpu().detach().numpy())
        edge_list['gene1'] = node1
        edge_list['gene2'] = node2
        save_dir = os.path.join(self.out_dir, "edge_mask")
        if not os.path.exists(save_dir): 
            print(f'{save_dir} does not exists, creating {save_dir}')
            os.makedirs(save_dir)
        edge_list.to_csv(os.path.join(save_dir, f'{out_name}.csv'), sep='\t', index=False)

    @property
    def output_results(self, ):
        if self.node_idx == []: return None

        for i in range(1, self.repeat + 1):
            out_name = f'{self.node}'
            out_name += '' if i == 1 and self.repeat == 1 else f'_{i}'
            feat_mask, edge_mask, edge_index, edge_list, pred_y = self.explain_node
            feat_mask = feat_mask.cpu().detach().numpy().reshape(1, -1)
            self.save_feat_mask(feat_mask, out_name)

            if edge_mask is not None:
                self.save_edge_mask(edge_index, edge_list, out_name)

            if self.viz_subgraph and (edge_mask is not None):
                self.generate_subgraph(edge_index, edge_mask, pred_y, out_name)

        return feat_mask


class Batch_Explain():
    def __init__(self, dataset, explainer, node_list, out_dir, normalized=False, patient=False, format='png') -> None:
        '''
        out_dir: 'explain/MCF7/repeat'
        normalized: {'sum', 'rank', False}
        '''
        self.dataset = dataset
        self.explainer = explainer
        self.node_list = node_list
        self.out_dir = out_dir
        self.normalized = normalized
        self.format = format

        if not patient:
            self.label = ['ATAC-1','CTCF-1','CTCF-2','CTCF-3','H3K4me3-1','H3K4me3-2','H3K27ac-1','H3K27ac-2','Means-SNV','Means-CNV']
        else:
            self.label = ['ATAC','CTCF', 'H3K27ac']
        for i in range(dataset[0].x.shape[1] - len(self.label)):
            self.label.append('HiC-{}'.format(i))
    
    def explain(self, viz_subgraph=True, repeat=1, repeat_times=5,):
        for node in self.node_list:
            single_task = Single_Node_Explain(self.dataset, self.explainer, node, self.out_dir, self.label, viz_subgraph, repeat, repeat_times, self.format)
            feat_mask = single_task.output_results

            if feat_mask is None:
                continue
            
            if self.normalized == 'sum':
                feat_mask = sum_norm(feat_mask)
            elif self.normalized == 'rank':
                feat_mask_arg = np.argsort(feat_mask, axis=0)
                for i in range(feat_mask.shape[0]):
                    feat_mask[feat_mask_arg[i]] = i

class Visualizer():
    def __init__(self, out_dir, fig_name, node_list, label) -> None:
        self.fig_name = fig_name
        self.out_dir = out_dir
        self.node_list = node_list
        self.label = label

    def feature_heatmap(self, all_feat_mask, format='png'):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks(range(len(self.label)))
        ax.set_yticklabels(self.label)
        ax.set_xticks(range(len(self.node_list)))
        ax.set_xticklabels(self.node_list)
        plt.xticks(rotation=45)
        im = ax.imshow(all_feat_mask, cmap=plt.cm.hot_r, aspect="auto")
        plt.colorbar(im)
        plt.savefig(os.path.join(self.out_dir, f"{self.fig_name}"+format), transparent=False, format=format)
    
    @property
    def draw_heatmap(self):
        all_feat_mask = None
        feat_dir = os.path.join(self.out_dir, 'feature_mask')
        for node in self.node_list:
            feat_mask = pd.read_csv(feat_dir + f'{node}.csv', sep='\t', index_col=0)
            print(feat_mask.columns)
            all_feat_mask = feat_mask if all_feat_mask is None else pd.concat([all_feat_mask, feat_mask], axis=0)
        
        all_feat_mask.to_csv(os.path.join(self.out_dir, f"{self.fig_name}.csv"), sep='\t')

        self.feature_heatmap(all_feat_mask, self.fig_name,)

def main(args, ckpt, configs, node_list, out_dir, format='png'): 
    '''
    ckpt: 'outs/MCF7_CPDB/best_model.pkl'
    '''
    dataset = get_data(configs)
    hidden_channels = configs['hidden_channels']
    heads = configs['heads']
    drop_rate = configs['ppi_drop_rate']
    attn_drop = configs['ppi_attn_drop']
    print("Loading model from {}".format(ckpt))
    model = CGMega(in_channels=dataset[0].num_node_features, hidden_channels=hidden_channels, heads=heads,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop, edge_dim=1, residual=True, devices_available=configs['gpu'])  
    model.load_state_dict(t.load(ckpt, map_location=t.device(configs['gpu']))['state_dict'])
    explainer = GATExplainer(model=model, epochs=500, num_hops=2, return_type='prob',)
    task = Batch_Explain(dataset, explainer, node_list, out_dir=out_dir, normalized='sum', patient=True if args.patient else False, format=format)
    drawer = Visualizer(node_list=node_list, out_dir=out_dir, fig_name='important_genes', label=task.label)

    if args.s:
        task.explain(viz_subgraph=True, repeat=5)

    elif args.heatmap:
        drawer.draw_heatmap()

    else:
        task.explain(viz_subgraph=True)

if __name__ == "__main__":
    while True:
        try:
            args = arg_parse()
            configs = config_load.get()
            gpu = f"cuda:{args.gpu}"
            configs["gpu"] = gpu
            if args.output:
                out_dir = args.output
                if not os.path.exists(out_dir):
                    print(f'{out_dir} do not exists, creating {out_dir}')
                    os.makedirs(out_dir)
            if args.patient:
                for patient in args.patient:
                    configs['data_dir'] = f'data/AML_Matrix/{patient}'
                    model_dir = f'outs/AML/{patient}'
                    ckpt = select_best_model(model_dir)
                    if not args.output:
                        out_dir = f'explain/AML/{patient}/predict_pos/' if not args.known else f'explain/AML/{patient}/true_pos/'
                    if args.node_list:
                        with open(args.node_list, mode='r') as f:
                            node_list = [node.strip() for node in f.readlines()]
                    else:
                        node_list = list(pd.read_csv(f'predict/AML/{patient}/{patient}_0.97+.csv')['gene_name']) if not args.known else get_labeled_nodes(cancer='AML', labels=[1], patient=patient)
                    main(args=args, ckpt=ckpt, configs=configs, node_list=node_list, out_dir=out_dir)
                sys.exit()
            if args.node_list:
                with open(args.node_list, mode='r') as f:
                    node_list = [node.strip() for node in f.readlines()]
            else:
                node_list = ['BRCA1']
            ckpt = 'outs/MCF7_CPDB/best_model.pkl' if not args.model else args.model
            if not args.output:
                out_dir = 'explain/MCF7_CPDB'
            main(args=args, ckpt=ckpt, configs=configs, node_list=node_list, out_dir=out_dir, format=args.format)
            
        except Exception as e: #fix ocassionally occur bug
            print(e)
            time.sleep(5)
            print("Retrying...")

            continue

        break