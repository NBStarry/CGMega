import re
import numpy as np
import pandas as pd
import random
import argparse
import os

import torch as t
import torch.nn
from torch_geometric.loader import NeighborLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from tqdm import tqdm
import wandb

import config_load
from data_preprocess_cv import get_data, get_cell_line, CancerDataset
from model import *
from utils import *


def arg_parse():
    parser = argparse.ArgumentParser(description="Train CGMega arguments.")
    parser.add_argument('-b', "--benchmark", dest="bm", action="store_true")
    parser.add_argument('-ch, "--change_hic', dest="change_hic_mat",
                        help="change origin Hi-C matrix", action="store_true")
    parser.add_argument('-cv, "--cross_validation', dest="cv",
                        help="use cross validation", action="store_true")
    parser.add_argument('-d', "--down_sample", dest="ds", action="store_true")
    parser.add_argument('-f', "--finetune", dest='finetune', help="finetune", action="store_true")
    parser.add_argument('-g', "--gpu", dest="gpu", default=None)
    parser.add_argument('-hg', "--hic_graph", dest="hic_graph", help="construct graph with hic", action="store_true")
    parser.add_argument('-hr', "--hic_reduce", dest="hic_reduce", help="change hic reduction method", action="store_true")
    parser.add_argument('-l', "--load", dest="load", help="load data", action="store_true")
    parser.add_argument('-p', "--predict", dest='pred', help="predict all nodes", action="store_true")
    parser.add_argument('--pan', dest='pan', default=False)
    parser.add_argument('-pt', "--patient", dest='patient', help="Patient ID", default=None, nargs='*')
    parser.add_argument('-r', "--reverse", dest="reverse", action='store_true')
    return parser.parse_args()


def drop_samples(dataset, fold, sample_neg=0., sample_pos=1., num_samples=0, random_seed=42):
    if sample_neg == 1 and sample_pos == 1:
        return []
    drop_neg = 1 - sample_neg
    drop_pos = 1 - sample_pos
    splitted_idx = dataset.get_idx_split(fold)
    train_idx = splitted_idx['train']
    drop_neg_idx, drop_pos_idx = [], []
    for i in train_idx:
        if dataset[0].y[i][0]:
            drop_neg_idx.append(i.item())
        if dataset[0].y[i][1]:
            drop_pos_idx.append(i.item())
    num_neg_samples = len(drop_neg_idx)
    num_pos_samples = len(drop_pos_idx)
    random.seed(random_seed)
    if num_samples:
        num_neg = int(num_samples * num_neg_samples /
                      (num_neg_samples + num_pos_samples))
        num_pos = num_samples - num_neg
        drop_neg_idx = random.sample(drop_neg_idx, num_neg_samples - num_neg)
        drop_pos_idx = random.sample(drop_pos_idx, num_pos_samples - num_pos)
    else:
        drop_neg_idx = random.sample(
            drop_neg_idx, int(num_neg_samples*drop_neg))
        drop_pos_idx = random.sample(
            drop_pos_idx, int(num_pos_samples*drop_pos))
    drop_idx = drop_neg_idx + drop_pos_idx
    print(
        f"Negatives: {num_neg_samples - len(drop_neg_idx)}, Positives: {num_pos_samples - len(drop_pos_idx)}")
    dataset[0].train_mask[drop_idx, fold] = False

    return drop_idx


def get_model(params, dataset):
    if params["model"] == "DualGATRes":
        model = DualGATRes2(dataset["ppi"].num_node_features, hidden_channels=params["hidden_channels"], heads=params["heads"],
                            drop_rate=params["drop_rate"], hic_attn_drop_rate=params['ppi_attn_drop'], ppi_attn_drop_rate=params['ppi_attn_drop'], edge_dim=dataset["hic"][0].edge_dim, devices_available=params["device"])
        
    elif params["model"] == "CGMega":
        model = CGMega(in_channels=dataset.num_node_features, hidden_channels=params['hidden_channels'], heads=params['heads'],
                    drop_rate=params['drop_rate'], attn_drop_rate=params['ppi_attn_drop'], edge_dim=dataset[0].edge_dim, residual=True, devices_available=params["device"])
    
    elif params["model"] == "GCN":
        model = GCN(in_channels=dataset.num_node_features, hidden_channels=params["hidden_channels"],
                    drop_rate=params["drop_rate"], devices_available=params["device"])
    
    elif params["model"] == "GAT":
        model = GAT(in_channels=dataset.num_node_features, hidden_channels=params["hidden_channels"], heads=params["heads"],
                    drop_rate=params["drop_rate"], attn_drop_rate=params['ppi_attn_drop'], edge_dim=dataset[0].edge_dim, devices_available=params["device"])
    
    elif "MLP" in params["model"]:
        model = MLP(in_channels=dataset.num_node_features, drop_rate=params["drop_rate"], devices_available=params["device"])
    
    elif "SVM" in params["model"]:
        model = SVM(C=1.0, gamma='scale', kernel='poly')
    
    elif params["model"] == "EMOGI":
        model = EMOGI(in_channels=dataset.num_node_features, drop_rate=0.5, pos_loss_multiplier=45.0, hidden_dims=[300, 100], devices_available=params["device"])

    elif params["model"] == "MTGCN":
        model = MTGCN(in_channels=dataset.num_node_features, hidden_dims=[300, 100], devices_available=params["device"])

    return model


def get_training_modules(params, dataset, pred=False):
    if params["model"] == "MTGCN":
        loss_func = torch.nn.BCEWithLogitsLoss()

    else:
        loss_func = torch.nn.BCELoss()

    fold = params["fold"]
    params["sample_neg"] = 1. if pred else params["sample_neg"]
    print("Drop ", 1 - params['sample_neg'], "of negative train samples and ", 1 - params['sample_pos'], "of positive train samples")
    if isinstance(dataset, dict):
        drop_idx = drop_samples(dataset["hic"], fold, sample_neg=
            params['sample_neg'], sample_pos=params['sample_pos'], random_seed=params['random_seed'])
    else:
        drop_idx = drop_samples(dataset, fold, sample_neg=
            params['sample_neg'], sample_pos=params['sample_pos'], random_seed=params['random_seed'])
    if params["sample_rate"] < 1:
        print("Drop ", (1 - params['sample_rate']), "of train samples")
        drop_idx += drop_samples(dataset, fold, params['sample_rate'], params['sample_rate'], params['random_seed'])
    elif params["sample_rate"] > 1: # 顺便当做采样数使用
        print(f"Sample {params['sample_rate']} train samples")
        drop_idx = drop_samples(dataset, fold, num_samples=params['sample_rate'], random_seed=params['random_seed'])
    neighbors = params['neighbors']
    if params["graph"] == "dual": params["model"] = "DualGATRes"
    data_list = [dataset["hic"][0], dataset["ppi"][0]] if params["graph"] == "dual" else [dataset[0]]
    model = get_model(params, dataset)

    loader_list, valid_loader_list, test_loader_list, unknown_loader_list = [], [], [], []
    for data in data_list:
        if pred:
            loader_list.append(NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                    input_nodes=data.train_mask[:, fold] + data.valid_mask[:, fold] + data.test_mask, shuffle=False if params["graph"]=="dual" else True))
            valid_loader_list, test_loader_list = None, None
        else:
            loader_list.append(NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                    input_nodes=data.train_mask[:, fold], shuffle=False if params["graph"]=="dual" else True))
            valid_loader_list.append(NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                        input_nodes=data.valid_mask[:, fold], shuffle=False if params["graph"]=="dual" else True))
            test_loader_list.append(NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                        input_nodes=data.test_mask))
            
        unknown_loader_list.append(NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                        input_nodes=data.unlabeled_mask))
    if params["model"] == "DualGATRes":
        optimizer = t.optim.AdamW([
            dict(params=model.extractor.parameters(), weight_decay=0.05),
            dict(params=model.discriminator.parameters(), weight_decay=0.05)
        ], lr=params['lr'])

    elif params["model"] == "EMOGI":
        opt_list = [dict(params=model.convs[0].parameters(), weight_decay=0.005)] + \
               [dict(params=model.convs[i].parameters(), weight_decay=0) for i in range(1, len(model.convs))]
        optimizer = t.optim.Adam(opt_list, lr=params['lr'])

    elif params["model"] == "CGMega":
        optimizer = t.optim.AdamW([
            dict(params=model.convs.parameters(), weight_decay=params['weight_decay']),
            dict(params=model.lins.parameters(), weight_decay=params['weight_decay'])
        ], lr=params['lr'])

    elif "SVM" in params["model"]:
        optimizer = t.optim.SGD(model.parameters(), lr=params['lr'])

    else:
        optimizer = t.optim.AdamW(model.parameters(), lr=params['lr'])

    num_training_steps = sum(
        data_list[0].train_mask[:, fold]) / params['batch_size'] * params['num_epochs']
    warmup_steps = 0.2 * num_training_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


    modules = {'dataset': dataset,
               'model': model,
               'loss_func': loss_func,
               'train_loader_list': loader_list,
               'valid_loader_list': valid_loader_list,
               'test_loader_list': test_loader_list,
               'unknown_loader_list': unknown_loader_list,
               'optimizer': optimizer,
               'scheduler': scheduler,
               'drop_idx': drop_idx}

    return modules


def calculate_metrics(y_true, y_pred, y_score):
    num_correct = np.equal(y_true, y_pred).sum()
    acc = (num_correct / y_true.shape[0])
    cf_matrix = confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=[0.0, 1.0])
    auprc = average_precision_score(y_true=y_true, y_score=y_score)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    return acc, cf_matrix, auprc, f1, auc


def score_avg_perfomance(train_result, score_col, logfile):
    folds = len(score_col)
    y_true = train_result['Label']
    y_score = train_result['avg_score']
    y_pred = train_result['pred_label']
    acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true.to_numpy(), y_pred.to_numpy(), y_score.to_numpy())
    tp = cf_matrix[1, 1]
    with open(logfile, 'a') as f:
        print(f"{folds}-folds AUPRC:{auprc:.4f}, AUROC:{auc:.4f}, ACC:{acc:.4f}, F1:{f1:.4f}, TP:{tp:.1f}",
              file=f, flush=True)


def pred_to_csv(configs, result):
    labeled = "known" if "Label" in result.columns else "unknown"
    cell_line = get_cell_line(configs["data_dir"])
    gene_list = get_gene_list()
    gene_list['gene_index'] = np.arange(gene_list.shape[0])
    result = gene_list.merge(result)
    result = result.drop('gene_index', axis=1)
    out_dir = f"./predict/result/best{cell_line}_{labeled}_result"
    out_dir += '_r' if configs['reverse'] else ''
    out_dir += ".csv" if configs["hic"] else "_wohic.csv"
    result.to_csv(out_dir, sep='\t', index=False)

def grid_search_SVM(model, dataset):
    train_mask = dataset[0].train_mask[:, 0]
    valid_mask = dataset[0].valid_mask[:, 0]

    x_train = dataset[0].x[train_mask + valid_mask]
    y_train = dataset[0].y[train_mask + valid_mask][:, 1]

    param_grid = {
        'C': [1/4, 1/2, 1, 2, 4],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(model.svm, param_grid, scoring='average_precision', cv=10, verbose=2)
    for _ in tqdm(range(1)):
        grid_search.fit(x_train, y_train)
    
    return grid_search.best_params_


def train_SVM(configs, dataset):
    model = SVM()
    best_params = grid_search_SVM(model, dataset)
    with open(configs['logfile'], 'a') as f:
        print("Best params:", best_params, file=f, flush=True)

    train_result = []
    test_mask = dataset[0].test_mask
    x_test = dataset[0].x[test_mask]
    y_test = dataset[0].y[test_mask][:, 1].numpy()
    index = dataset[0].pos[test_mask].numpy()
    for fold in range(configs['cv_folds']):
        model = SVM(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
        train_mask = dataset[0].train_mask[:, fold]
        x_train = dataset[0].x[train_mask]
        y_train = dataset[0].y[train_mask][:, 1]

        model.svm.fit(x_train, y_train)
        y_score = model.svm.predict_proba(x_test)[:, 1]
        y_pred = np.zeros_like(y_score)
        y_pred[y_score>0.5] = 1
        train_result =  pred_to_df(fold, train_result, index, y_test, y_score)

    score_col = [f"score_{i}" for i in range(configs['cv_folds'])]
    train_result['avg_score'] = train_result[score_col].mean(axis=1)
    train_result['pred_label'] = train_result.apply(
        lambda x: 1 if x['avg_score'] > 0.5 else 0, axis=1)
    score_avg_perfomance(train_result, score_col, configs['logfile'])


def train(model, fold, train_loader_list, valid_loader_list, optimizer, devices, scheduler=None, loss_func=None,):
    model.train()
    tot_loss = 0
    acc = 0
    data = None
    out = None
    steps = 0

    for data in zip(*train_loader_list):
        steps = steps + 1
        optimizer.zero_grad()
        size = data[0].batch_size
        out = model(data)
        if isinstance(model, MTGCN):
            out, rl, c1, c2 = out[0][:size], out[1], out[2], out[3]
        else: out = out[:size]
        true_lab = data[0].y[:size][:, 1].to(devices)
        out = out.view(-1)
        loss = loss_func(out, true_lab.float())
        if isinstance(model, MTGCN):
            loss = loss / (c1 ** 2) + rl / (c2 ** 2) + 2 * torch.log(c1 * c2)
        del out, true_lab
        loss.backward()
        tot_loss = tot_loss + loss.item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    tot_loss = tot_loss/steps

    model.eval()
    y_true = np.array([])
    y_score = np.array([])
    train_correct = 0
    num_train = 0
    for data in zip(*train_loader_list):
        size = data[0].batch_size
        with torch.no_grad():
            out = model(data)
            if isinstance(model, MTGCN):
                out, rl, c1, c2 = out[0][:size], out[1], out[2], out[3]
            else: out = out[:size]
        true_lab = data[0].y[:size][:, 1].to(devices)
        out = out.view(-1)
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        train_correct += t.eq(pred_lab, true_lab).sum().float()
        num_train += size
        train_mask = data[0].train_mask[:size, fold]
        if y_score.size == 0:
            y_score = out[train_mask[:size]].cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_true = np.append(y_true, true_lab.cpu().detach().numpy())

    train_acc = (train_correct / num_train).cpu().detach().numpy()
    train_auprc = average_precision_score(y_true=y_true, y_score=y_score)
    train_auc = roc_auc_score(y_true, y_score)
    print(f"Train AUPRC: {train_auprc:.4f}, AUROC: {train_auc:.4f}, ACC: {train_acc:.4f}")

    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    valid_loss = 0
    steps = 0
    for data in zip(*valid_loader_list):
        steps = steps + 1
        size = data[0].batch_size
        with torch.no_grad():
            out = model(data)
            if isinstance(model, MTGCN):
                out, rl, c1, c2 = out[0][:size], out[1], out[2], out[3]
            else: out = out[:size]
        true_lab = data[0].y[:size][:, 1].to(devices)
        out = out.view(-1)
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        y_pred = np.append(y_pred, pred_lab.cpu().detach().numpy())
        if y_score.size == 0:
            y_score = out.cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_true = np.append(y_true, true_lab.cpu().detach().numpy())
        valid_loss = valid_loss + loss_func(out, true_lab.float()).item()

    valid_loss = valid_loss / steps
    acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)

    return tot_loss, valid_loss, acc, cf_matrix, auprc, f1, auc, train_auprc, train_auc, train_acc


def train_model(modules, params, log_name, fold, head_info=False):
    fold = params["fold"]
    logfile = params['logfile']
    devices = params["device"]
    dataset = modules['dataset']
    model = modules['model']
    loader_list = modules['train_loader_list']
    valid_loader_list = modules['valid_loader_list']
    optimizer = modules['optimizer']
    scheduler = modules['scheduler']
    loss_func = modules['loss_func']

    if isinstance(dataset, dict):
        data = dataset["ppi"][0]
    else:
        data = dataset[0]
    if head_info:
        config_load.print_config(logfile, params)
        with open(logfile, 'a') as f:
            print("Model: CGMega\nTrain/Valid/Test: ",
                  data.train_mask[:, fold].sum(), data.valid_mask[:, fold].sum(), data.test_mask.sum(),
                  file=f, flush=True)

    if head_info:
        with open(logfile, 'a') as f:
            print(model, file=f, flush=True)

    if params['wandb']:
        wandb.init(project=params['project'], config=params,
                   settings=wandb.Settings(start_method='fork'))
        wandb.watch_called = False
        wandb.watch(model, log="all")
    print('Start Training')
    vmax_auprc = 0
    trigger_times = 0
    for epoch in range(params['num_epochs']):
        train_loss, valid_loss, acc, cf_matrix, auprc, f1, auc, train_auprc, train_auc, train_acc = train(model, fold, loader_list,
                                                                                                          valid_loader_list,
                                                                                                          optimizer, devices, scheduler,
                                                                                                          loss_func)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Acc: {acc:.4f}, " \
                  f"Auprc: {auprc:.4f}, TP: {cf_matrix[1, 1]}, F1: {f1:.4f}, Auroc: {auc:.4f}")
        if params['wandb']:
            wandb.log({
                "loss_epoch": train_loss,
                "loss": valid_loss,
                "Acc": acc,
                "AUPRC": auprc,
                "True_negative:": cf_matrix[0, 0],
                "False_positive": cf_matrix[0, 1],
                "False_negative": cf_matrix[1, 0],
                "True_positive": cf_matrix[1, 1],
                "auc": auc,
                "f1 score": f1,
                "train_auprc": train_auprc,
                "train_auc": train_auc,
                "train_acc": train_acc,
            })

        if epoch >= params["num_epochs"] // 10:
            if auprc < vmax_auprc:
                trigger_times += 1
                if trigger_times == params["num_epochs"] // 5:
                    print("Early Stopping")
                    break
            else:
                trigger_times = 0
                vmax_auprc = auprc
                max_epoch = epoch
                best_acc = acc
                best_tp = cf_matrix[1, 1]
                best_auc = auc
                best_f1 = f1
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
    if not os.path.exists(os.path.join(params['out_dir'], log_name)):
        os.mkdir(os.path.join(params['out_dir'], log_name))
    if params['wandb']:
        run_name = re.findall(r'\d+', str(wandb.run.name))[0]
        model_dir = os.path.join(params['out_dir'], log_name, \
            run_name + f'_{fold}_{vmax_auprc:.4f}_{best_tp}.pkl')
    else:
        model_dir = os.path.join(params['out_dir'], log_name, \
            f'{fold}_{vmax_auprc:.4f}_{best_auc:.4f}_{best_tp}.pkl')
    t.save(checkpoint, model_dir)
    with open(logfile, 'a') as f:
        if params['wandb']:
            print("{} epoch {}: AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}".format(
                run_name, max_epoch, vmax_auprc, best_auc, best_acc, best_f1, best_tp), file=f, flush=True)
        else:
            print("epoch {}: AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}".format(
                max_epoch, vmax_auprc, best_auc, best_acc, best_f1, best_tp), file=f, flush=True)
    if isinstance(dataset, dict):
        dataset["hic"][0].train_mask[modules['drop_idx'], fold] = True
    else:
        dataset[0].train_mask[modules['drop_idx'], fold] = True
    if params['wandb']:
        wandb.finish()
    return vmax_auprc, best_auc, best_acc, best_f1, best_tp, model_dir


def predict(model, loader_list, params, ckpt, labeled=True):
    devices = params["device"]
    print(f"Loading model from {ckpt} ......")
    model.load_state_dict(t.load(ckpt)['state_dict'])
    model.eval()

    y_true = np.array([]) if labeled else None

    y_pred = np.array([])
    y_score = np.array([])
    y_index = np.array([])

    for data in zip(*loader_list):
        size = data[0].batch_size
        with t.no_grad():
            out = model(data)
            out = out[0][:size] if isinstance(model, MTGCN) else out[:size]
        out = t.squeeze(out)
        index = data[0].pos[:size]
        true_lab = data[0].y[:size][:, 1].to(devices) if labeled else None
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        if y_score.size == 0:
            y_score = out.cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_pred = np.append(y_pred, pred_lab.cpu().detach().numpy())
        y_index = np.append(y_index, index.cpu().detach().numpy())

        y_true = np.append(y_true, true_lab.cpu().detach().numpy()) if labeled else None


    return y_score, y_pred, y_true, y_index


def pred_to_df(i, result, y_index, y_true, y_score):
    if i == 0:
        mid = np.array([y_index, y_true, y_score]).T if y_true is not None else np.array([y_index, y_score]).T
        result = pd.DataFrame(data=mid, columns=['gene_index', 'Label',
                                    f'score_{i}'] if y_true is not None else ['gene_index',
                                    f'score_{i}'])
    else:
        mid = np.array([y_index, y_score]).T
        mid = pd.DataFrame(data=mid, columns=['gene_index', f'score_{i}'])
        result = result.merge(mid)

    return result


def cv_train(args, configs, disturb=None):
    if args.finetune:
        ckpt_path = f"./predict/models/{configs['model']}"
        if configs['model'] == 'CGMega':
            if 'AML' in configs['data_dir']:
                ckpt = ckpt_path + ('/K562_Hi-C/527_0.8551_38.pkl')
            else:
                ckpt = ckpt_path + ('/MCF7_Hi-C/328_0.8976_24.pkl' if configs["hic"] else '/MCF7_PPI/78_0.9815_27.pkl') 
        elif configs['model'] == 'MTGCN':
            ckpt = ckpt_path + '/0.8936_0.9614_22.pkl'
        elif configs['model'] == 'GCN':
            ckpt = ckpt_path + '/0.7302_0.8995_18.pkl'
        elif configs['model'] == 'GAT':
            ckpt = ckpt_path + '/0.8837_0.9570_23.pkl'
    log_name = configs['log_name']
    num_folds = configs["cv_folds"]
    dataset = get_data(configs=configs, stable=configs["stable"]) if disturb is None else get_data(configs=configs, stable=configs["stable"], disturb_list=disturb)
    if 'SVM' in configs['model']:
        train_SVM(configs, dataset)
        return
    sum_auprc, sum_auc, sum_acc, sum_f1, sum_tp, train_result = [], [], [], [], [], []
    for j in range(configs['repeat']):
        head_info = True if j == 0 else False
        if args.cv:
            for i in range(num_folds):
                if i != 0:
                    head_info = False
                configs["fold"] = i
                modules = get_training_modules(configs, dataset)
                if args.finetune:
                    print(f"Loading model from {ckpt} ......")
                    modules["model"].load_state_dict(t.load(ckpt)['state_dict'])
                auprc, auc, acc, f1, tp, new_ckpt = train_model(
                    modules, configs, log_name, i, head_info,)
                sum_auprc.append(auprc)
                sum_auc.append(auc)
                sum_acc.append(acc)
                sum_f1.append(f1)
                sum_tp.append(tp)
                y_score, y_pred, y_true, y_index = predict(modules['model'], modules['test_loader_list'],
                                                                                    configs, new_ckpt)
                acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)
                tp = cf_matrix[1, 1]
                with open(configs['logfile'], 'a') as f:
                    print("Test AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}"
                        .format(auprc, auc, acc, f1, tp), file=f, flush=True)
                train_result =  pred_to_df(i, train_result, y_index, y_true, y_score)
            score_col = [f"score_{i}" for i in range(num_folds)]
            train_result['avg_score'] = train_result[score_col].mean(axis=1)
            train_result['pred_label'] = train_result.apply(
                lambda x: 1 if x['avg_score'] > 0.5 else 0, axis=1)
            score_avg_perfomance(train_result, score_col, configs['logfile'])
        else:
            configs["fold"] = 7
            modules = get_training_modules(configs, dataset)
            if args.finetune:
                print(f"Loading model from {ckpt} ......")
                modules["model"].load_state_dict(t.load(ckpt)['state_dict'])
            auprc, auc, acc, f1, tp, new_ckpt = train_model(
                modules, configs, log_name, configs['fold'], head_info)
            head_info = False
            sum_auprc.append(auprc)
            sum_auc.append(auc)
            sum_acc.append(acc)
            sum_f1.append(f1)
            sum_tp.append(tp)
    with open(configs['logfile'], 'a') as f:
        print("Avg AUPRC:{:.4f}±{:.4f}, AUC:{:.4f}±{:.4f}, ACC:{:.4f}±{:.4f}, F1:{:.4f}±{:.4f}, TP:{:.1f}±{:.1f}".format(np.average(sum_auprc), np.std(sum_auprc),
                                                                                                                         np.average(sum_auc), np.std(sum_auc), np.average(sum_acc), np.std(sum_acc), np.average(sum_f1), np.std(sum_f1), np.average(sum_tp), np.std(sum_tp)), file=f, flush=True)


def predict_all(args, configs):
    dataset = get_data(configs=configs)
    num_folds = configs["cv_folds"]
    ckpt_path = "./predict/models/MCF7_Hi-C/" if not args.patient else f"./predict/models/AML/{configs['patient']}"
    ckpt_path += '_r/' if args.reverse else '/'
    checkpoint = os.listdir(ckpt_path)
    known_result, unknown_result = [], []
    for i in range(num_folds):
        configs["fold"] = i
        modules = get_training_modules(configs, dataset, pred=True)
        y_score, y_pred, y_true, y_index = predict(modules['model'], modules['train_loader_list'],
                                                    configs, ckpt_path + checkpoint[i])
        known_result =  pred_to_df(i, known_result, y_index, y_true, y_score)
        y_score, y_pred, y_true, y_index = predict(modules['model'], modules['unknown_loader_list'],
                                                    configs, ckpt_path + checkpoint[i], labeled=False)
        unknown_result = pred_to_df(i, unknown_result, y_index, y_true, y_score)

    score_col = [f"score_{i}" for i in range(configs["cv_folds"])]
    known_result['avg_score'] = known_result[score_col].mean(axis=1)
    known_result['pred_label'] = known_result.apply(
        lambda x: 1 if x['avg_score'] > 0.5 else 0, axis=1)
    unknown_result['avg_score'] = unknown_result[score_col].mean(axis=1)
    unknown_result['pred_label'] = unknown_result.apply(
        lambda x: 1 if x['avg_score'] > 0.5 else 0, axis=1)
    
    pred_to_csv(configs, known_result)
    pred_to_csv(configs, unknown_result)


def down_sample_migrate(args, configs):
    configs["data_dir"] = "data/Leukemia_Matrix"
    configs["project"] = "Finetune"
    num_samples = [i for i in range(1000, 99, -100)]
    num_samples.extend([i for i in range(90, 9, -20)])
    num_samples.append(250)
    for model in ['GCN', 'GAT', 'MTGCN']:
        configs['model'] = model
        for seed in [40, 41, 42, 43, 44]:
            for num in num_samples:
                configs['hic'] = True
                configs['sample_rate'] = num
                configs['log_name'] = f"/migrate/finetune/{configs['model']}/sample_test/{num}_{seed}"
                configs['logfile'] = configs['log_dir'] + configs['log_name'] + '.txt'
                configs['random_seed'] = seed
                cv_train(args, configs)


def down_sample_train(args, configs):
    configs['data_dir'] = "data/Leukemia_Matrix"
    sample_list = [(1, 0.33), (0.67, 0.43), (0.5, 0.49), (0.4, 0.52), (0.33, 0.54), (0.28, 0.56), (0.25, 0.57), (0.22, 0.58), (0.2, 0.59)]
    for model in ['GCN', 'GAT', 'MTGCN', 'EMOGI']:
        configs['model'] = model
        for sample_rate in sample_list:
            configs['sample_pos'] = sample_rate[0]
            configs['sample_neg'] = sample_rate[1]
            configs['log_name'] = f"k562_{model}_sample_{sample_rate[0]}_{sample_rate[1]}"
            configs['logfile'] = os.path.join(configs['log_dir'], configs['log_name'] + '.txt')
            cv_train(args, configs)


def hic_graph(args, configs):
    configs["stable"] = False
    configs["graph"] = "dual" # "dual", "onlyc", "plusc"
    for norm in ['binary']:
        configs["hic_norm"] = norm
        configs["log_name"] = f"mcf7_hic_{norm}_{configs['graph']}_graph"
        configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
        configs["ppi"] = "CPDB"
        configs["load_data"] = True
        cv_train(args, configs)


def ways_of_reduction(args, configs):
    configs["stable"] = False
    configs["hic_reduce"] = "t-sne"
    for i in [1, 2, 3]:
        configs["hic_reduce_dim"] = i
        configs["log_name"] = "mcf7_hic_" + configs["hic_reduce"] + f"_{i}"
        configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
        cv_train(args, configs)


def run_benchmark(args, configs):
    for ppi in ["IRef", "PCNet", "STRING", "Multinet"]:
        configs["ppi"] = ppi
        for model in ["EMOGI", "MTGCN", "GCN", "GAT"]: # 'N2V_MLP', 'N2V_SVM'
            configs["model"] = model
            configs["log_name"] = f"mcf7_{model}" if configs['hic'] else f"mcf7_{model}(woH)"
            configs["log_name"] += f"_{ppi}" if configs["ppi"] != "CPDB" else ""
            configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
            disturb = {'add': ['PPI']} if 'N2V' in model else None
            cv_train(args, configs, disturb)


def patient_train(args, configs):
    print(args.patient)
    for patient in args.patient:
        configs['data_dir'] = f'data/AML_Matrix/{patient}'
        configs["log_name"] = f"{configs['data_dir'].split('/')[-1]}"
        configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
        cv_train(args, configs)


def pan_train(args, configs):
    print(f"Training {args.pan} dataset.")
    for ppi in ['STRING', 'irefindex', 'Mentha', 'BioPlex', 'CPDB_v34', 'HINT', 'HumanNet', 'InBioMap', 'IntAct']:  
        configs['ppi'] = ppi
        configs['data_dir'] = f'data/{args.pan}'
        configs['stable'] = False
        configs['pan'] = True
        configs['cv_folds'] = 5
        configs['log_name'] = f"{configs['data_dir'].split('/')[-1]}_{configs['model']}_{configs['ppi']}"
        configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
        configs['lr'] = 0.0001
        configs['neighbors'] = [40,10]
        cv_train(args, configs)

def main(args, configs):
    if args.pred:
        for patient in args.patient:
            configs['patient'] = patient
            configs['data_dir'] = f'data/AML_Matrix/{patient}'
            predict_all(args, configs)

    elif args.hic_graph:
        hic_graph(args, configs)

    elif args.hic_reduce:
        ways_of_reduction(args, configs)

    elif args.change_hic_mat:
        configs["stable"] = False
        for i in [1, 2, 5, 10]:
            configs["hic_reduce_dim"] = i
            configs["log_name"] = "mcf7_hic_" + configs["hic_type"] + f"_{i}"
            configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
            cv_train(args, configs)

    elif args.bm:
        run_benchmark(args, configs)

    elif args.patient:
        patient_train(args, configs)

    elif args.pan:
        pan_train(args, configs)

    elif args.ds:
        down_sample_train(args, configs)

    else:
        configs["log_name"] = f"{get_cell_line(configs['data_dir'])[1:]}_{configs['ppi']}"
        configs["logfile"] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
        cv_train(args, configs)


if __name__ == "__main__":
    configs = config_load.get()
    args = arg_parse()
    gpu = f"cuda:{args.gpu}" if args.gpu else 'cpu'
    configs["device"] = gpu
    configs['load_data'] = args.load
    if args.reverse:
        configs["reverse"] = True
    main(args, configs)