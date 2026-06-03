
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class HeteroGNNEncoder(nn.Module):


    def __init__(self, metadata, hidden_dim=64, num_layers=2):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:  # edge_types
                src_type, _, dst_type = edge_type

                conv_dict[edge_type] = SAGEConv(
                    in_channels=(-1, -1) if i > 0 else (metadata[0][src_type], metadata[0][dst_type]),
                    out_channels=hidden_dim,
                    aggr='mean'
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

    def forward(self, x_dict, edge_index_dict):

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # ReLU
            x_dict = {key: F.relu(v) for key, v in x_dict.items()}
            # Dropout
            x_dict = {key: F.dropout(v, training=self.training) for key, v in x_dict.items()}

        h_graph = x_dict['company']

        return h_graph