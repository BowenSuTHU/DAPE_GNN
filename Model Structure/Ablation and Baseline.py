import torch
from torch import nn
import math
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import SAGEConv
from torch import Tensor

class NodeEmbedding(nn.Module):
    def __init__(self,d_vocab,d_model):

        super(NodeEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(d_vocab,d_model,device='cpu')

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)


class GNNlayer(nn.Module):
    def __init__(self,d_model,d_hidden,dropout1,d_vocab,device,heads = 8):
        super(GNNlayer,self).__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.embedding = nn.Embedding(d_vocab,d_model,scale_grad_by_freq = True)
        self.Leakyrelu = nn.LeakyReLU(0.15)
        self.gatlayer_y = GATConv(d_model, d_hidden,concat=False, heads=heads, dropout=dropout1)
        self.gatlayer_y1 = GATConv(d_hidden, d_hidden,concat=False, heads=heads, dropout=dropout1)
        self.layernorm = nn.LayerNorm(d_hidden,elementwise_affine=True,device=device)
        self.layernorm0 = nn.LayerNorm(d_model,elementwise_affine=False,device=device)
        self.mlp_out = nn.Linear(d_hidden,d_vocab)
        self.MLP_y = nn.Linear(d_hidden, d_hidden)
        self.gelu_y = nn.GELU()
        self.MLP_y1 = nn.Linear(d_hidden, d_hidden)
        self.gelu_y1 = nn.GELU()
        self.d_hidden = d_hidden
    def forward(self, edge_index, N,y):
        y0 = self.embedding(y)
        y = y0.view(-1,self.d_model)
        y = self.layernorm0(y)
        y = self.gatlayer_y(y,edge_index)
        y1 = self.Leakyrelu(y)
        y1 = self.MLP_y(y1)
        y1 = self.gelu_y(y1)
        y2 = self.gatlayer_y1(y1,edge_index)
        y2 = self.MLP_y1(y2)
        y2 = self.gelu_y1(y2)
        y2 = y2.reshape(1,-1,self.d_hidden)
        trg_y = y2[:,N,:]
        trg = self.mlp_out(trg_y)
        out = trg.view(-1,self.d_vocab)
        return out

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self,input_size :int,max_seq_len :int,
                 dim_val: int,
                 n_encoder_layers: int,
                 n_heads: int,
                 dropout_encoder,
                 dim_feedforward_encoder: int,
                 dropout_pos_enc
                 ):
        super(Encoder,self).__init__()
        self.n_heads = n_heads
        self.dim_val = dim_val
        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc,
                                                           max_len=max_seq_len)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)

        self.fc1 = nn.Sequential(
        nn.Linear(32, 16),
        nn.LeakyReLU(0.2),

        nn.Linear(16, 16),
        nn.LeakyReLU(0.2),

        nn.Linear(16, 14))
        self.fc2 = nn.Sequential(
        nn.Linear(16, 16),
        nn.LeakyReLU(0.2),
        nn.Linear(16, 8),
        nn.LeakyReLU(0.2),
        nn.Linear(8, 1)
        )
        self.fc0 = nn.Sequential(
             nn.Linear(1024,512),
             nn.GELU(),
             nn.Linear(512,206),
             nn.GELU(),
             nn.Linear(206,50),
             nn.GELU(),
             nn.Linear(50,1),
             nn.GELU()
             )
    def forward(self,src:Tensor,freq,temp) -> Tensor:
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        x = self.encoder(src)
        x = self.fc0(x.view(x.shape[0], 32, 1024))
        x = x.view(x.shape[0],32)
        x = self.fc1(x)
        y = self.fc2(torch.cat((x, freq, temp), 1))
        return y

class GCNlayer(nn.Module):
    def __init__(self,d_model,d_hidden,d_vocab,device):
        super(GCNlayer,self).__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.embedding = nn.Embedding(d_vocab,d_model,scale_grad_by_freq = True)
        self.Leakyrelu = nn.LeakyReLU(0.15)
        self.layernorm0 = nn.LayerNorm(d_model,elementwise_affine=False,device=device)
        self.gcnlayer1 = GCNConv(d_model, d_hidden)
        self.gcnlayer2 = GCNConv(d_hidden, d_hidden)
        self.mlp_out = nn.Linear(d_hidden,d_vocab)
        self.d_hidden = d_hidden
    def forward(self, edge_index, N,y):
        y0 = self.embedding(y)
        y = y0.view(-1,self.d_model)
        y = self.layernorm0(y)
        y = self.gcnlayer1(y,edge_index)
        y1 = self.Leakyrelu(y)
        y2 = self.gcnlayer2(y1,edge_index)
        y2 = y2.reshape(1,-1,self.d_hidden)
        trg_y = y2[:,N,:]
        trg = self.mlp_out(trg_y)
        out = trg.view(-1,self.d_vocab)
        return out

class GINlayer(nn.Module):
    def __init__(self,d_model,d_hidden,d_vocab,device):
        super(GINlayer,self).__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.embedding = nn.Embedding(d_vocab,d_model,scale_grad_by_freq = True)
        self.Leakyrelu = nn.LeakyReLU(0.15)
        self.layernorm0 = nn.LayerNorm(d_model,elementwise_affine=False,device=device)
        self.gcnlayer1 = GINConv(nn.Linear(d_model,d_hidden))
        self.gcnlayer2 = GINConv(nn.Linear(d_hidden,d_hidden))
        self.mlp_out = nn.Linear(d_hidden,d_vocab)
        self.d_hidden = d_hidden
    def forward(self, edge_index, N,y):
        y0 = self.embedding(y)
        y = y0.view(-1,self.d_model)
        y = self.layernorm0(y)
        y = self.gcnlayer1(y,edge_index)
        y1 = self.Leakyrelu(y)
        y2 = self.gcnlayer2(y1,edge_index)
        y2 = y2.reshape(1,-1,self.d_hidden)
        trg_y = y2[:,N,:]
        trg = self.mlp_out(trg_y)
        out = trg.view(-1,self.d_vocab)
        return out

class SAGElayer(nn.Module):
    def __init__(self,d_model,d_hidden,d_vocab,device):
        super(SAGElayer,self).__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.embedding = nn.Embedding(d_vocab,d_model,scale_grad_by_freq = True)
        self.Leakyrelu = nn.LeakyReLU(0.15)

        self.layernorm0 = nn.LayerNorm(d_model,elementwise_affine=False,device=device)
        self.gcnlayer1 = SAGEConv(d_model, d_hidden)
        self.gcnlayer2 = SAGEConv(d_hidden, d_hidden)
        self.mlp_out = nn.Linear(d_hidden,d_vocab)
        self.d_hidden = d_hidden
    def forward(self, edge_index, N,y):
        y0 = self.embedding(y)
        y = y0.view(-1,self.d_model)
        y = self.layernorm0(y)
        y = self.gcnlayer1(y,edge_index)
        y1 = self.Leakyrelu(y)
        y2 = self.gcnlayer2(y1,edge_index)
        y2 = y2.reshape(1,-1,self.d_hidden)
        trg_y = y2[:,N,:]
        trg = self.mlp_out(trg_y)
        out = trg.view(-1,self.d_vocab)
        return out
