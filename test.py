import os
import torch as t
import tqdm

from data_preprocess_cv import get_data, CancerDataset
import config_load
from main import *
from utils import get_node_name

configs = config_load.get()
gpu = f"cuda:0"
configs["device"] = gpu
configs['load_data'] = True
configs["log_name"] = f"test"
configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
configs["batch_size"] = 2000
# configs["hic"] = False
# configs['data_dir'] = "data/Leukemia_Matrix/"

dataset = get_data(configs=configs, stable=True)
ckpt_path = "./predict/models/MCF7_Hi-C/" if configs['hic'] else './predict/models/MCF7_PPI/'
checkpoints = os.listdir(ckpt_path)
train_result = []
for i in range(10):
    configs["fold"] = i
    modules = get_training_modules(configs, dataset, pred=True)
    ckpt = ckpt_path + checkpoints[i]
    # ckpt = t.load(ckpt)['state_dict']
    y_score, y_pred, y_true, y_index, edge_weight_1, edge_weight_2, edge_index_1, edge_index_2 = predict(modules['model'], modules['train_loader_list'], configs, ckpt)
    # print(edge_weight_1)
    data_list = []
    for k in range(len(edge_weight_1)):
        for j in tqdm(range(len(edge_weight_1[k]))):
            data_list.append(
                [y_index[edge_index_1[k][1][j].item()],
                y_index[edge_index_1[k][0][j].item()],
                edge_weight_1[k][j][0].item(),
                edge_weight_1[k][j][1].item(),
                edge_weight_1[k][j][2].item(),
                edge_weight_2[k][j][0].item(),
                edge_weight_2[k][j][1].item(),
                edge_weight_2[k][j][2].item(),
                ])
            # if j == 100: break
    edge_attention = pd.DataFrame(data_list, columns=['Gene 1', 'Gene 2', 'Attention 1-1', 'Attention 1-2', 'Attention 1-3', 'Attention 2-1', 'Attention 2-2', 'Attention 2-3'])
        # break
    edge_attention['Gene 1'] = get_node_name(edge_attention['Gene 1'].tolist())
    edge_attention['Gene 2'] = get_node_name(edge_attention['Gene 2'].tolist())
    edge_attention.to_excel(f"data/Results/Edge_Attention_{i}.xlsx", index=False)