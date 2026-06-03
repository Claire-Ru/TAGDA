
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDivider(nn.Module):

    def __init__(self, input_dim, shared_dim, private_dim, topo_dim=8):

        super().__init__()

        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.topo_dim = topo_dim

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim * 2),
            nn.BatchNorm1d(shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim * 2, shared_dim),
        )

        self.private_encoder = nn.Sequential(
            nn.Linear(input_dim, private_dim * 2),
            nn.BatchNorm1d(private_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(private_dim * 2, private_dim),
        )

        self.graph_reconstructor = nn.Linear(
            shared_dim + private_dim, input_dim
        )

        self.topo_reconstructor = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, topo_dim),
            nn.Sigmoid()
        )

    def forward(self, h_graph):

        h_shared = self.shared_encoder(h_graph)
        h_private = self.private_encoder(h_graph)
        return h_shared, h_private

    def compute_orthogonality_loss(self, h_shared, h_private):

        h_shared_norm = F.normalize(h_shared, dim=-1)
        h_private_norm = F.normalize(h_private, dim=-1)
        inner_product = torch.matmul(h_shared_norm.t(), h_private_norm)
        orth_loss = torch.norm(inner_product, p='fro') ** 2
        return orth_loss

    def compute_reconstruction_loss(self, h_graph, h_shared, h_private,
                                    s_topo=None, lambda_topo=0.5):

        loss_detail = {}

        h_concat = torch.cat([h_shared, h_private], dim=-1)   # [N, shared+private]
        h_recon = self.graph_reconstructor(h_concat)           # [N, input_dim]
        rec_graph_loss = F.mse_loss(h_recon, h_graph)
        loss_detail['rec_graph_loss'] = rec_graph_loss.item()

        if s_topo is not None:
            s_recon = self.topo_reconstructor(h_shared)        # [N, topo_dim]
            rec_topo_loss = F.mse_loss(s_recon, s_topo)
            loss_detail['rec_topo_loss'] = rec_topo_loss.item()
        else:
            rec_topo_loss = torch.tensor(0.0, device=h_graph.device)
            loss_detail['rec_topo_loss'] = 0.0

        rec_loss = rec_graph_loss + lambda_topo * rec_topo_loss
        loss_detail['rec_total_loss'] = rec_loss.item()

        return rec_loss, loss_detail