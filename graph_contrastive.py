
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphContrastiveModule(nn.Module):

    def __init__(self, shared_dim=48, proj_dim=32,
                 temperature=0.1, noise_std=0.05, drop_rate=0.1, top_k=5,
                 max_contrast_samples=1024):
        super().__init__()

        self.temperature = temperature
        self.noise_std = noise_std
        self.drop_rate = drop_rate
        self.top_k = top_k
        self.max_contrast_samples = max_contrast_samples


        self.projector = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, proj_dim)
        )


    def _augment(self, h):
        if not self.training:
            return h
        noise = torch.randn_like(h) * self.noise_std
        h_aug = h + noise
        h_aug = F.dropout(h_aug, p=self.drop_rate, training=True)
        return h_aug

    def intra_contrastive_loss(self, h_shared):

        N = h_shared.size(0)
        if N < 2:
            return torch.tensor(0.0, device=h_shared.device)


        if N > self.max_contrast_samples:
            idx = torch.randperm(N, device=h_shared.device)[:self.max_contrast_samples]
            h_shared = h_shared[idx]
            N = self.max_contrast_samples

        z1 = F.normalize(self.projector(self._augment(h_shared)), dim=-1)  # [N, proj_dim]
        z2 = F.normalize(self.projector(self._augment(h_shared)), dim=-1)  # [N, proj_dim]

        z = torch.cat([z1, z2], dim=0)

        sim = torch.mm(z, z.t()) / self.temperature

        eye_mask = torch.eye(2 * N, dtype=torch.bool, device=h_shared.device)
        sim.masked_fill_(eye_mask, float('-inf'))

        labels = torch.cat([
            torch.arange(N, 2 * N, device=h_shared.device),
            torch.arange(0, N, device=h_shared.device)
        ])  # [2N]

        loss = F.cross_entropy(sim, labels)
        return loss

    def cross_topo_contrastive_loss(self, h_shared_src, s_topo_src,
                                     h_shared_tgt, s_topo_tgt):

        N_src = h_shared_src.size(0)
        N_tgt = h_shared_tgt.size(0)

        if N_src < 2 or N_tgt < 2:
            return torch.tensor(0.0, device=h_shared_src.device)

        if N_src > self.max_contrast_samples:
            idx_s = torch.randperm(N_src, device=h_shared_src.device)[:self.max_contrast_samples]
            h_shared_src = h_shared_src[idx_s]
            s_topo_src = s_topo_src[idx_s]
            N_src = self.max_contrast_samples
        if N_tgt > self.max_contrast_samples:
            idx_t = torch.randperm(N_tgt, device=h_shared_tgt.device)[:self.max_contrast_samples]
            h_shared_tgt = h_shared_tgt[idx_t]
            s_topo_tgt = s_topo_tgt[idx_t]
            N_tgt = self.max_contrast_samples

        z_src = F.normalize(self.projector(h_shared_src), dim=-1)  # [N_src, proj_dim]
        z_tgt = F.normalize(self.projector(h_shared_tgt), dim=-1)  # [N_tgt, proj_dim]

        topo_src_norm = F.normalize(s_topo_src.float(), dim=-1)
        topo_tgt_norm = F.normalize(s_topo_tgt.float(), dim=-1)

        topo_sim = torch.mm(topo_src_norm, topo_tgt_norm.t())

        feat_sim = torch.mm(z_src, z_tgt.t()) / self.temperature

        k = min(self.top_k, N_src)

        _, pos_idx = topo_sim.topk(k, dim=0)

        loss = torch.tensor(0.0, device=h_shared_src.device)

        for j in range(N_tgt):

            pos_mask = torch.zeros(N_src, device=h_shared_src.device)
            pos_mask[pos_idx[:, j]] = 1.0

            logits_j = feat_sim[:, j]  # [N_src]

            loss += F.binary_cross_entropy_with_logits(
                logits_j, pos_mask
            )

        return loss / N_tgt

    @staticmethod
    def entropy_min_loss(logits_tgt):

        probs = F.softmax(logits_tgt, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean()


    def forward(self, h_shared_src, h_shared_tgt,
                s_topo_src=None, s_topo_tgt=None,
                logits_tgt=None,
                lambda_intra=0.5,
                lambda_cross=1.0,
                lambda_entropy=0.1):

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=h_shared_src.device)

        intra_src = self.intra_contrastive_loss(h_shared_src)
        intra_tgt = self.intra_contrastive_loss(h_shared_tgt)
        intra_loss = (intra_src + intra_tgt) / 2.0
        loss_dict['intra_contrastive_loss'] = intra_loss.item()
        total_loss = total_loss + lambda_intra * intra_loss

        if s_topo_src is not None and s_topo_tgt is not None:
            cross_loss = self.cross_topo_contrastive_loss(
                h_shared_src, s_topo_src,
                h_shared_tgt, s_topo_tgt
            )
            loss_dict['cross_topo_contrastive_loss'] = cross_loss.item()
            total_loss = total_loss + lambda_cross * cross_loss
        else:
            loss_dict['cross_topo_contrastive_loss'] = 0.0

        if logits_tgt is not None:
            ent_loss = self.entropy_min_loss(logits_tgt)
            loss_dict['entropy_min_loss'] = ent_loss.item()
            total_loss = total_loss + lambda_entropy * ent_loss
        else:
            loss_dict['entropy_min_loss'] = 0.0

        loss_dict['total_contrastive_loss'] = total_loss.item()
        return total_loss, loss_dict