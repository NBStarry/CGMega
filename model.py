import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, GCNConv, GATConv, ChebConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
from torch.nn import Dropout, MaxPool1d

from sklearn.svm import SVC

import config_load

HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2
configs = config_load.get()
HIC_DIM = configs["hic_reduce_dim"]


def freeze(layer):
    for child in layer.children():
        for p in child.parameters():
            p.requires_grad = False


class SVM(t.nn.Module):
    def __init__(self, C=1.0, gamma='auto', kernel='rbf'):
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)

    def forward(self, data):
        data = data[0]
        return self.svm.predict(data.x)


class MLP(t.nn.Module):
    def __init__(self, drop_rate, in_channels, devices_available):
        super(MLP, self).__init__()
        self.devices_available = devices_available

        self.dropout = Dropout(drop_rate)
        self.lins = t.nn.ModuleList()
        self.lins.append(
            Linear(in_channels, HIDDEN_DIM, weight_initializer="kaiming_uniform").to(devices_available))
        self.lins.append(
            Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform").to(devices_available))

    def forward(self, x):
        x = x[0].to(self.devices_available).x if isinstance(x, tuple) else x
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)
        return t.sigmoid(x)


class CGMega(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, residual, devices_available):
        super(CGMega, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate
        self.convs = t.nn.ModuleList()
        self.residual = residual
        mid_channels = in_channels + hidden_channels if residual else hidden_channels

        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim,
                                          concat=False, beta=True).to(self.devices_available))
        self.convs.append(TransformerConv(mid_channels, hidden_channels, heads=heads,
                                          dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True).to(self.devices_available))
        
        self.ln1 = LayerNorm(in_channels=mid_channels).to(self.devices_available)
        self.ln2 = LayerNorm(in_channels=hidden_channels *
                             heads).to(self.devices_available)
        
        self.pool = MaxPool1d(2, 2)

        self.dropout = Dropout(drop_rate)
        self.lins = t.nn.ModuleList()
        self.lins.append(
            Linear(int(hidden_channels*heads/2), HIDDEN_DIM, weight_initializer="kaiming_uniform").to(devices_available))
        self.lins.append(
            Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform").to(devices_available))
        

    def forward(self, data):
        data = data[0].to(self.devices_available)
        x = data.x
        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        res = x
        x, weight_1 = self.convs[0](x, edge_index, edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.ln1(x)

        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        x, weight_2 = self.convs[1](x.to(self.devices_available), edge_index.to(
            self.devices_available), edge_attr.to(self.devices_available), return_attention_weights=True)
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)
        x = t.unsqueeze(x, 1)
        x = self.pool(x)
        x = t.squeeze(x)
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)

        return t.sigmoid(x), weight_1, weight_2


class GATModule(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, devices_available):
        super(GATModule, self).__init__()
        self.drop_rate = drop_rate
        self.convs = t.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim,
                            concat=False, beta=True).to(devices_available))
        self.ln1 = LayerNorm(in_channels=hidden_channels +
                             in_channels).to(devices_available)
        self.convs.append(TransformerConv(hidden_channels + in_channels, hidden_channels, heads=heads,
                                          dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True).to(
            devices_available))
        self.ln2 = LayerNorm(in_channels=hidden_channels *
                             heads).to(devices_available)

    def forward(self, data):
        data = data.to(self.devices_available)
        x = data.x
        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        res = x
        x = self.convs[0](x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1)
        x = self.ln1(x)
        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        x = self.convs[1](x.to(self.devices_available), edge_index.to(self.devices_available),
                          edge_attr.to(self.devices_available))
        return x


class Extractor(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, hic_attn_drop_rate, ppi_attn_drop_rate, edge_dim, devices_available):
        super(Extractor, self).__init__()
        self.drop_rate = drop_rate
        self.convs = t.nn.ModuleList()
        ppi_module = GATModule(in_channels, hidden_channels,
                               heads, drop_rate, ppi_attn_drop_rate, edge_dim)
        hic_module = GATModule(in_channels, hidden_channels,
                               heads, drop_rate, hic_attn_drop_rate, edge_dim)
        self.convs.append(hic_module)
        self.convs.append(ppi_module)
        self.poolhic = MaxPool1d(2, 2)
        self.poolppi = MaxPool1d(2, 2)
        self.lins = t.nn.ModuleList()
        self.lins.append(Linear(int(hidden_channels*heads/2), HIC_DIM,
                         weight_initializer="kaiming_uniform").to(devices_available))
        self.lins.append(Linear(int(hidden_channels*heads/2), int(hidden_channels*heads/2),
                         weight_initializer="kaiming_uniform").to(devices_available))
        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)

    def forward(self, data):
        hic_data, ppi_data = data[0], data[1]
        batch_size = hic_data.batch_size
        hic_x = self.convs[0](hic_data)[:batch_size]
        ppi_x = self.convs[1](ppi_data)[:batch_size]
        hic_x = t.unsqueeze(hic_x, 1)
        ppi_x = t.unsqueeze(ppi_x, 1)
        hic_x = self.poolhic(hic_x)
        ppi_x = self.poolppi(ppi_x)
        hic_x = t.squeeze(hic_x)
        ppi_x = t.squeeze(ppi_x)
        hic_x = self.lins[0](hic_x).relu()
        hic_x = self.dropout1(hic_x)
        ppi_x = self.lins[1](ppi_x).relu()
        ppi_x = self.dropout2(ppi_x)
        x = t.cat((ppi_x, hic_x), dim=1)
        return x


class DualGATRes2(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, hic_attn_drop_rate, ppi_attn_drop_rate, edge_dim, devices_available):
        super(DualGATRes2, self).__init__()
        self.extractor = Extractor(in_channels, hidden_channels, heads,
                                   drop_rate, hic_attn_drop_rate, ppi_attn_drop_rate, edge_dim)
        self.dicriminator = MLP(drop_rate, HIC_DIM + int(hidden_channels*heads/2), devices_available)

    def forward(self, data):
        x = self.extractor(data)
        return self.dicriminator(x)


class GCN(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, drop_rate, devices_available):
        super(GCN, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate

        self.convs = t.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, improved=False).to(self.devices_available))
        self.convs.append(GCNConv(hidden_channels, 1, improved=False).to(self.devices_available))

        # self.lins = MLP(drop_rate, hidden_channels*heads, devices_available)

    def forward(self, data):
        data = data[0].to(self.devices_available)
        x = data.x
        x = self.convs[0](x, data.edge_index).relu()
        x = self.convs[1](x.to(self.devices_available), data.edge_index.to(
            self.devices_available))
    
        return t.sigmoid(x)


class GAT(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, devices_available):
        super(GAT, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate

        self.convs = t.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim, concat=False).to(self.devices_available))
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim, concat=True).to(self.devices_available))

        self.lins = MLP(drop_rate, hidden_channels*heads, devices_available)
        

    def forward(self, data):
        data = data[0].to(self.devices_available)
        x = data.x
        x = self.convs[0](x, data.edge_index, data.edge_attr)
        x = self.convs[1](x.to(self.devices_available), data.edge_index.to(
            self.devices_available), data.edge_attr.to(self.devices_available))

        return self.lins(x)
    
# according to https://github.com/Bibyutatsu/proEMOGI/blob/main/proEMOGI/proemogi.py
class EMOGI(t.nn.Module):
    def __init__(self, in_channels, devices_available, num_hidden_layers=2, drop_rate=0.5,
                 hidden_dims=[20, 40], pos_loss_multiplier=1, weight_decay=5e-4,):
        super(EMOGI, self).__init__()
        self.in_channels = in_channels
        self.devices_available = devices_available

        # model params
        self.weight_decay = weight_decay
        self.pos_loss_multiplier = pos_loss_multiplier
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims
        self.drop_rate = drop_rate
        
        self.convs = t.nn.ModuleList()

        # add intermediate layers
        inp_dim = self.in_channels
        for l in range(self.num_hidden_layers):
            self.convs.append(GCNConv(inp_dim,
                                       self.hidden_dims[l]).to(self.devices_available))
            inp_dim = self.hidden_dims[l]
            
        self.convs.append(GCNConv(self.hidden_dims[-1], 1).to(self.devices_available))
        
    def forward(self, data):
        data = data[0].to(self.devices_available)
        x = data.x
        for layer in self.convs[:-1]:
            x = layer(x, data.edge_index)
            x = F.relu(x)
            if self.drop_rate is not None:
                x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        return t.sigmoid(x)
    
# according to https://github.com/weiba/MTGCN/blob/master/MTGCN.py
class MTGCN(t.nn.Module):
    def __init__(self, in_channels, hidden_dims, devices_available):
        super(MTGCN, self).__init__()
        self.devices_available = devices_available

        self.conv1 = ChebConv(in_channels, hidden_dims[0], K=2, normalization="sym").to(self.devices_available)
        self.conv2 = ChebConv(hidden_dims[0], hidden_dims[1], K=2, normalization="sym").to(self.devices_available)
        self.conv3 = ChebConv(hidden_dims[1], 1, K=2, normalization="sym").to(self.devices_available)

        self.lin1 = Linear(in_channels, 100).to(self.devices_available)
        self.lin2 = Linear(in_channels, 100).to(self.devices_available)

        self.c1 = t.nn.Parameter(t.Tensor([0.5])).to(self.devices_available)
        self.c2 = t.nn.Parameter(t.Tensor([0.5])).to(self.devices_available)

    def forward(self, data):
        data = data[0].to(self.devices_available)
        edge_index, _ = dropout_adj(data.edge_index, p=0.5, force_undirected=True, training=self.training)

        x0 = F.dropout(data.x, training=self.training)
        x = t.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = t.relu(self.conv2(x, edge_index))

        x = x1 + t.relu(self.lin1(x0))
        z = x1 + t.relu(self.lin2(x0))

        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)
        E = data.edge_index

        pos_loss = -t.log(t.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()
        neg_edge_index = negative_sampling(pb, data.x.shape[0], data.edge_index.shape[1])
        neg_loss = -t.log(
            1 - t.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean() if neg_edge_index.numel() != 0 else 0

        r_loss = pos_loss + neg_loss


        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return (x, r_loss, self.c1, self.c2)