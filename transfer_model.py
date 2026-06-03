
import torch
import torch.nn as nn
import torch.nn.functional as F
from zuizhonng_c.model.hetero_gnn_encoder import HeteroGNNEncoder
from zuizhonng_c.model.feature_divider import FeatureDivider
from zuizhonng_c.model.domain_alignment import ClassConditionalAlignment, coral_loss
from zuizhonng_c.model.graph_contrastive import GraphContrastiveModule


class TransferLearningModel(nn.Module):

    def __init__(self, metadata, hidden_dim=64, num_layers=2,
                 shared_dim=48, private_dim=16, num_classes=2,
                 use_mmd=True, topo_dim=8,
                 proj_dim=32, temperature=0.1,
                 noise_std=0.05, drop_rate=0.1, top_k=5):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.num_classes = num_classes
        self.topo_dim = topo_dim

        self.gnn_encoder = HeteroGNNEncoder(
            metadata=metadata,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        self.feature_divider = FeatureDivider(
            input_dim=hidden_dim,
            shared_dim=shared_dim,
            private_dim=private_dim,
            topo_dim=topo_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim + private_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.domain_alignment = ClassConditionalAlignment(
            num_classes=num_classes,
            use_mmd=use_mmd,
            shared_dim=shared_dim,
            private_dim=private_dim
        )

        self.graph_contrastive = GraphContrastiveModule(
            shared_dim=shared_dim,
            proj_dim=proj_dim,
            temperature=temperature,
            noise_std=noise_std,
            drop_rate=drop_rate,
            top_k=top_k
        )

    def forward(self, x_dict, edge_index_dict, return_features=False):

        h_graph = self.gnn_encoder(x_dict, edge_index_dict)

        h_shared, h_private = self.feature_divider(h_graph)

        h_concat = torch.cat([h_shared, h_private], dim=-1)
        logits = self.classifier(h_concat)

        if return_features:
            return {
                'logits':    {'company': logits},
                'h_graph':   {'company': h_graph},
                'h_shared':  {'company': h_shared},
                'h_private': {'company': h_private}
            }
        else:
            return logits

    def compute_loss(self, outputs_src, labels_src, outputs_tgt, train_mask_src,
                     train_mask_tgt,
                     s_src=None, s_tgt=None,
                     lambda_orth=0.01, lambda_rec=0.1, lambda_align=1.0,
                     lambda_topo=0.5,
                     lambda_contrast=1.0,
                     lambda_intra=0.5,
                     lambda_cross=1.0,
                     lambda_entropy=0.1,
                     progress=0.0):

        device = outputs_src['logits']['company'].device
        loss_dict = {}

        cls_loss_src = self.compute_loss_task(
            outputs_src['logits']['company'], labels_src, train_mask_src, domain='source'
        )
        loss_dict['cls_loss_src'] = cls_loss_src.item()

        cls_loss_tgt = torch.tensor(0.0, device=device)
        loss_dict['cls_loss_tgt'] = 0.0

        orth_loss = self.feature_divider.compute_orthogonality_loss(
            outputs_src['h_shared']['company'][train_mask_src],
            outputs_src['h_private']['company'][train_mask_src]
        )
        loss_dict['orth_loss'] = orth_loss.item()

        s_src_train = None
        if s_src is not None:
            s_src_train = s_src[train_mask_src]   # [N_train, topo_dim]

        rec_loss, rec_detail = self.feature_divider.compute_reconstruction_loss(
            h_graph=outputs_src['h_graph']['company'][train_mask_src],
            h_shared=outputs_src['h_shared']['company'][train_mask_src],
            h_private=outputs_src['h_private']['company'][train_mask_src],
            s_topo=s_src_train,
            lambda_topo=lambda_topo
        )
        loss_dict.update(rec_detail)

        align_loss = torch.tensor(0.0, device=device)

        if train_mask_src is not None and train_mask_src.sum() > 0:
            if train_mask_tgt is None or train_mask_tgt.sum() == 0:

                h_src = outputs_src['h_shared']['company'][train_mask_src]
                h_tgt = outputs_tgt['h_shared']['company']

                coral_loss_val = coral_loss(h_src, h_tgt)
                loss_dict['coral_loss'] = coral_loss_val.item()
                align_loss += coral_loss_val

                domain_align_loss, domain_dict = self.domain_alignment(
                    h_shared_src=outputs_src['h_shared']['company'][train_mask_src],
                    labels_src=labels_src[train_mask_src],
                    h_shared_tgt=outputs_tgt['h_shared']['company'],
                    labels_tgt=torch.zeros(
                        outputs_tgt['h_shared']['company'].size(0),
                        dtype=torch.long, device=device
                    ),
                    h_private_src=outputs_src['h_private']['company'][train_mask_src],
                    h_private_tgt=outputs_tgt['h_private']['company'],
                    progress=progress
                )
                align_loss += domain_align_loss
                loss_dict.update(domain_dict)

        contrastive_loss = torch.tensor(0.0, device=device)

        if train_mask_src is not None and train_mask_src.sum() > 0:

            s_topo_src_all = s_src
            s_topo_tgt_all = s_tgt

            h_shared_src_train = outputs_src['h_shared']['company'][train_mask_src]
            s_topo_src_train = s_topo_src_all[train_mask_src] if s_topo_src_all is not None else None

            h_shared_tgt_all = outputs_tgt['h_shared']['company']

            logits_tgt_all = outputs_tgt['logits']['company']

            contrastive_loss, contrast_dict = self.graph_contrastive(
                h_shared_src=h_shared_src_train,
                h_shared_tgt=h_shared_tgt_all,
                s_topo_src=s_topo_src_train,
                s_topo_tgt=s_topo_tgt_all,
                logits_tgt=logits_tgt_all,
                lambda_intra=lambda_intra,
                lambda_cross=lambda_cross,
                lambda_entropy=lambda_entropy
            )
            loss_dict.update(contrast_dict)

        total_loss = (cls_loss_src
                      + lambda_orth     * orth_loss
                      + lambda_rec      * rec_loss
                      + lambda_align    * align_loss
                      + lambda_contrast * contrastive_loss)

        loss_dict['contrastive_loss_weighted'] = (lambda_contrast * contrastive_loss).item()
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def compute_loss_task(self, logits, labels, mask, domain='source'):
        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = logits[mask]
        labels = labels[mask]

        num_pos = (labels == 1).sum().item()
        num_neg = (labels == 0).sum().item()
        total   = num_pos + num_neg

        if num_pos > 0 and num_neg > 0:
            weight_pos = total / (2 * num_pos)
            weight_neg = total / (2 * num_neg)
            class_weights = torch.tensor([weight_neg, weight_pos], device=logits.device)
        else:
            class_weights = None

        return F.cross_entropy(logits, labels, weight=class_weights)