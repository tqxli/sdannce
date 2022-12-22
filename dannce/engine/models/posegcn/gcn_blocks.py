from inspect import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from dannce.engine.models.posegcn.normalization import get_normalization

"""
BASIC GCN CONV LAYERS
"""
class SemGraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        #very useful demo means this is Parameter, which can be adjust by bp methods
        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)

        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class ModulatedGraphConv(SemGraphConv):
    """
    https://github.com/ZhimingZo/Modulated-GCN/blob/main/Modulated_GCN/Modulated-GCN_benchmark/models/modulated_gcn_conv.py
    """
    def __init__(self, in_features, out_features, adj, bias=True):
        super(ModulatedGraphConv, self).__init__(in_features, out_features, adj, bias)

        # weight modulation matrix
        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        # affinity modulation matrix ("add" mode in the paper)
        self.adj2 = nn.Parameter(torch.ones_like(adj))        
        nn.init.constant_(self.adj2, 1e-6)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        
        adj = self.adj.to(input.device) + self.adj2.to(input.device)
        adj = (adj.T + adj)/2 # symmetry
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        
        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

"""
MODEL BLOCKS
"""
class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None, base_block='sem', norm_type="batch"):
        super(_GraphConv, self).__init__()

        if base_block == 'sem':
            convblock = SemGraphConv 
        elif base_block == 'modulated':
            convblock = ModulatedGraphConv

        self.gconv = convblock(input_dim, output_dim, adj)
        self.norm_type = norm_type
        self.bn = get_normalization(norm_type, output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # breakpoint()
        x = self.gconv(x).contiguous()

        if self.norm_type == "batch":
            x = x.transpose(1, 2).contiguous()
            x = self.bn(x).transpose(1, 2).contiguous()
        else:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _GraphConv_no_bn(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv_no_bn, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2).contiguous()
        return x

class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout, base_block='sem', norm_type="batch"):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout, base_block=base_block, norm_type=norm_type)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout, base_block=base_block, norm_type=norm_type)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out

class Node_Attention(nn.Module):
    def __init__(self,channels):
        '''
        likely SElayer
        '''
        super(Node_Attention,self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.squeeze = nn.Sequential(
            nn.Linear(channels,channels//4),
            nn.ReLU(),
            nn.Linear(channels//4,12),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.avg(x).squeeze(2)
        out = self.squeeze(out)
        out = out[:,None,:]
        out = out
        out = (x+x*out)
        return out

class _ResGraphConv_Attention(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv_Attention, self).__init__()

        self.gconv1 = _GraphConv_no_bn(adj, input_dim, hid_dim//2, p_dropout)


        self.gconv2 = _GraphConv_no_bn(adj, hid_dim//2, output_dim, p_dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.attention = Node_Attention(output_dim)

    def forward(self, x,joint_features=None):
        if joint_features is None:
            residual = x
        else:
            joint_features = joint_features.transpose(1,2).contiguous()
            x = torch.cat([joint_features,x],dim=2)
            residual = x
        # breakpoint()
        out = self.gconv1(x)
        out = self.gconv2(out)

        # out = self.bn(residual.transpose(1,2).contiguous() + out)
        # out = self.relu(out)

        # out = self.attention(out).transpose(1,2).contiguous()
        return out.transpose(1, 2) + residual
        #return torch.concat((out.transpose(1, 2), residual), dim=-1)

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(channels_in, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, channels_out)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

class GraphConv(nn.Module):
    
    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        #self.adj_sq = adj_sq
        self.activation = activation
        #self.scale_identity = scale_identity
        #self.I = Parameter(torch.eye(number_of_nodes, requires_grad=False).unsqueeze(0))


    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L
    
    
    def laplacian_batch(self, A_hat):
        #batch, N = A.shape[:2]
        #if self.adj_sq:
        #    A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        #I = torch.eye(N).unsqueeze(0).to(device)
        #I = self.I
        #if self.scale_identity:
        #    I = 2 * I  # increase weight of self connections
        #A_hat = A + I
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L


    def forward(self, X, A):
        batch = X.size(0)
        #A = self.laplacian(A)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        #X = self.fc(torch.bmm(A_hat, X))
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphPool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphPool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)


    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class GraphUnpool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphUnpool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)


    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class GraphUNet(nn.Module):

    def __init__(self, in_features=3, out_features=3):
        super(GraphUNet, self).__init__()

        self.A_0 = Parameter(torch.eye(23).float().cuda(), requires_grad=True)
        self.A_1 = Parameter(torch.eye(11).float().cuda(), requires_grad=True)
        self.A_2 = Parameter(torch.eye(5).float().cuda(), requires_grad=True)
        self.A_3 = Parameter(torch.eye(2).float().cuda(), requires_grad=True)
        self.A_4 = Parameter(torch.eye(1).float().cuda(), requires_grad=True)
        # self.A_5 = Parameter(torch.eye(1).float().cuda(), requires_grad=True)

        self.gconv1 = GraphConv(in_features, 4)
        self.pool1 = GraphPool(23, 11)

        self.gconv2 = GraphConv(4, 8)
        self.pool2 = GraphPool(11, 5)

        self.gconv3 = GraphConv(8, 16)
        self.pool3 = GraphPool(5, 2)

        self.gconv4 = GraphConv(16, 32)
        self.pool4 = GraphPool(2, 1)

        # self.gconv5 = GraphConv(32, 64)  # 2 = 1 H + 1 O
        # self.pool5 = GraphPool(2, 1)

        self.fc1 = nn.Linear(32, 20)

        self.fc2 = nn.Linear(20, 32)

        # self.unpool6 = GraphUnpool(1, 2)
        # self.gconv6 = GraphConv(64, 32)

        self.unpool7 = GraphUnpool(1, 2)
        self.gconv7 = GraphConv(64, 16)

        self.unpool8 = GraphUnpool(2, 5)
        self.gconv8 = GraphConv(32, 8)

        self.unpool9 = GraphUnpool(5, 11)
        self.gconv9 = GraphConv(16, 4)

        self.unpool10 = GraphUnpool(11, 23)
        self.gconv10 = GraphConv(8, out_features, activation=None)

        self.ReLU = nn.ReLU()

    def _get_decoder_input(self, X_e, X_d):
        return torch.cat((X_e, X_d), 2)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_0)
        X_1 = self.pool1(X_0)

        X_1 = self.gconv2(X_1, self.A_1)
        X_2 = self.pool2(X_1)

        X_2 = self.gconv3(X_2, self.A_2)
        X_3 = self.pool3(X_2)

        X_3 = self.gconv4(X_3, self.A_3)
        X_4 = self.pool4(X_3)

        # X_4 = self.gconv5(X_4, self.A_4)
        # X_5 = self.pool5(X_4)

        global_features = self.ReLU(self.fc1(X_4))
        global_features = self.ReLU(self.fc2(global_features))

        # X_6 = self.unpool6(global_features)
        # X_6 = self.gconv6(self._get_decoder_input(X_4, X_6), self.A_4)

        X_7 = self.unpool7(global_features)
        X_7 = self.gconv7(self._get_decoder_input(X_3, X_7), self.A_3)

        X_8 = self.unpool8(X_7)
        X_8 = self.gconv8(self._get_decoder_input(X_2, X_8), self.A_2)

        X_9 = self.unpool9(X_8)
        X_9 = self.gconv9(self._get_decoder_input(X_1, X_9), self.A_1)

        X_10 = self.unpool10(X_9)
        X_10 = self.gconv10(self._get_decoder_input(X_0, X_10), self.A_0)

        return X_10


class GraphNet(nn.Module):
    
    def __init__(self, in_features=2, out_features=2):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(29).float().cuda(), requires_grad=True)
        
        self.gconv1 = GraphConv(in_features, 128)
        self.gconv2 = GraphConv(128, 16)
        self.gconv3 = GraphConv(16, out_features, activation=None)
        
    
    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        X_2 = self.gconv3(X_1, self.A_hat)
        
        return 
