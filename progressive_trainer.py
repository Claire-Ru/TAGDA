
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, roc_curve
)
from pathlib import Path
from torch_geometric.utils import degree, to_undirected

EMA_DECAY = 0.9999


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_dynamic_ema_decay(epoch, total_epochs):
    progress = epoch / total_epochs
    return 0.99 + 0.0099 * progress


def get_structural_features(data, include_advanced=True):
    if hasattr(data['company'], 'topo_feat'):
        return data['company'].topo_feat

    # 回退：现场提取
    from zuizhonng_c.model.graph_alignment import extract_topology_features
    return extract_topology_features(data, include_advanced=include_advanced)



class ProgressiveTransferTrainer:

    def __init__(self, model, source_data, target_data, device='cuda'):
        self.model = model.to(device)
        self.source_data = source_data.to(device)
        self.target_data = target_data.to(device)
        self.device = device
        self.model.source_data = self.source_data
        self.model.target_data = self.target_data

        self.target_ratios = torch.tensor([0.754, 0.246], device=device)

        # Masks
        self.source_train_mask = source_data['company'].train_mask
        self.target_train_mask = target_data['company'].train_mask
        self.target_val_mask = target_data['company'].val_mask
        self.target_test_mask = target_data['company'].test_mask
        self.target_unlabeled_mask = self.target_train_mask.clone()

        print(f"  train: {self.target_train_mask.sum().item()} companies")
        print(f"  val: {self.target_val_mask.sum().item()} companies")
        print(f"  test: {self.target_test_mask.sum().item()} companies")

        Path('saved_models').mkdir(parents=True, exist_ok=True)
        self.best_model_state = None

        train_labels = source_data['company'].y[self.source_train_mask].cpu().numpy()
        self.train_pos_ratio = np.mean(train_labels == 1)


    def update_teacher(self, epoch=None, total_epochs=None):

        if epoch is not None and total_epochs is not None:
            ema_decay = get_dynamic_ema_decay(epoch, total_epochs)
        else:
            ema_decay = EMA_DECAY

        for param_t, param_s in zip(self.teacher_model.parameters(), self.model.parameters()):
            if param_t.data.size() != param_s.data.size():
                continue
            param_t.data = ema_decay * param_t.data + (1 - ema_decay) * param_s.data

    def _calculate_h_measure(self, y_true, y_score):
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            pi1 = np.mean(y_true == 1)
            pi0 = 1 - pi1
            c1 = pi0 * fpr
            c0 = pi1 * (1 - tpr)
            h_measure = 1 - np.min(c1 + c0)
            return float(h_measure)
        except:
            return 0.0

    def _evaluate(self, data, test_mask, domain='target', use_bayesian_threshold=True):
        self.model.eval()

        with torch.no_grad():
            logits = self.model(data.x_dict, data.edge_index_dict)

            if test_mask.sum() == 0:
                return {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'f1': 0.0, 'auc': 0.0, 'auprc': 0.0,
                    'f1_pos': 0.0, 'precision_pos': 0.0, 'recall_pos': 0.0,
                    'h_measure': 0.0,
                    'specificity': 0.0,
                    'sensitivity': 0.0
                }

            true = data['company'].y[test_mask].cpu().numpy()
            prob = F.softmax(logits, dim=-1)[test_mask, 1].cpu().numpy()

            if use_bayesian_threshold:
                quantile_threshold = np.percentile(prob, (1 - self.train_pos_ratio) * 100)
                threshold = max(0.3, min(0.7, quantile_threshold))
                pred = (prob > threshold).astype(int)
            else:
                pred = (prob > 0.5).astype(int)

            acc = accuracy_score(true, pred)
            p, r, f1, _ = precision_recall_fscore_support(
                true, pred, average='weighted', zero_division=0
            )

            p_class, r_class, f1_class, _ = precision_recall_fscore_support(
                true, pred, average=None, zero_division=0, labels=[0, 1]
            )

            if len(p_class) >= 2:
                precision_pos = p_class[1]
                recall_pos = r_class[1]
                f1_pos = f1_class[1]
            else:
                precision_pos = recall_pos = f1_pos = 0.0

            auc = roc_auc_score(true, prob) if len(np.unique(true)) > 1 else 0.0
            auprc = average_precision_score(true, prob) if len(np.unique(true)) > 1 else 0.0

            h_measure = self._calculate_h_measure(true, prob)

            tn = np.sum((true == 0) & (pred == 0))  # True Negative (observed good, classified good)
            fp = np.sum((true == 0) & (pred == 1))  # False Positive (observed good, classified bad)
            fn = np.sum((true == 1) & (pred == 0))  # False Negative (observed bad, classified good)
            tp = np.sum((true == 1) & (pred == 1))  # True Positive (observed bad, classified bad)

            # Sensitivity = TP / (TP + FN) = number of both observed bad and classified as bad / number of observed bad
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Specificity = TN / (TN + FP) = number of both observed good and classified as good / number of observed good
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
            'auc': auc,
            'auprc': auprc,
            'f1_pos': f1_pos,
            'precision_pos': precision_pos,
            'recall_pos': recall_pos,
            'h_measure': h_measure,
            'specificity': specificity,
            'sensitivity': sensitivity
        }

    def stage1_source_pretrain(self, epochs):

        print("\n" + "=" * 50)
        print("Stage 1")
        print("=" * 50)

        def reset_model_parameters(model):

            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        reset_model_parameters(self.model)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=5e-4
        )

        best_val_auc = 0.0
        self.best_model_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            logits = self.model(
                self.source_data.x_dict,
                self.source_data.edge_index_dict
            )

            train_mask = self.source_train_mask
            loss = F.cross_entropy(
                logits[train_mask],
                self.source_data['company'].y[train_mask]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()


            if epoch % 50 == 0 or epoch == epochs:
                metrics = self._evaluate(
                    self.target_data,
                    self.target_val_mask,
                    domain='target_val',
                    use_bayesian_threshold=True
                )
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                      f"Target Val AUC: {metrics['auc']:.4f}")

                if metrics['auc'] > best_val_auc:
                    best_val_auc = metrics['auc']
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

        print("Stage 1 completed")


        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"load Stage 1 best model")

        test_metrics = self._evaluate(
            self.target_data,
            self.target_test_mask,
            domain='target_test',
            use_bayesian_threshold=True
        )
        self._print_metrics("Source-Only", test_metrics)

        return test_metrics

    def stage2_target_finetune(self, epochs):

        print("\n" + "=" * 50)
        print("Stage 2")
        print("=" * 50)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=50,
            num_training_steps=epochs
        )

        s_src = get_structural_features(self.source_data)
        s_tgt = get_structural_features(self.target_data)

        best_val_auc = 0.0
        self.best_model_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            progress = epoch / epochs

            outputs_src = self.model(self.source_data.x_dict, self.source_data.edge_index_dict, return_features=True)
            outputs_tgt = self.model(self.target_data.x_dict, self.target_data.edge_index_dict, return_features=True)


            total_loss, loss_dict = self.model.compute_loss(
                outputs_src=outputs_src,
                outputs_tgt=outputs_tgt,
                labels_src=self.source_data['company'].y,
                train_mask_src=self.source_train_mask,
                train_mask_tgt=None,
                s_src=s_src,
                s_tgt=s_tgt,
                lambda_orth=0.01,
                lambda_rec=0.1,
                lambda_align=2.0 * (1 - progress) + 0.5 * progress,
                lambda_topo=0.5,
                lambda_contrast=0.5 * (1 - progress) + 0.1 * progress,
                lambda_intra=0.5,
                lambda_cross=1.0,
                lambda_entropy=0.1,
                progress=progress
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 50 == 0 or epoch == epochs:
                metrics = self._evaluate(self.target_data, self.target_val_mask, domain='target_val')
                omega = loss_dict.get('omega', 0.0)
                domain_loss = loss_dict.get('total_domain_loss', 0.0)
                cross_loss = loss_dict.get('cross_topo_contrastive_loss', 0.0)
                intra_loss = loss_dict.get('intra_contrastive_loss', 0.0)
                entropy_loss = loss_dict.get('entropy_min_loss', 0.0)

                print(
                    f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} | "
                    f"Val AUC: {metrics['auc']:.4f} | ω: {omega:.3f} | "
                    f"D: {domain_loss:.4f} | "
                    f"Cross: {cross_loss:.4f} | Intra: {intra_loss:.4f} | Ent: {entropy_loss:.4f}"
                )

                if metrics['auc'] > best_val_auc:
                    best_val_auc = metrics['auc']
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        for _ in range(10):
            self.update_teacher()

        test_metrics = self._evaluate(self.target_data, self.target_test_mask, domain='target_test')
        self._print_metrics("Stage2", test_metrics)

        from pathlib import Path
        save_path = Path('./outputs/models')
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'teacher_state_dict': self.teacher_model.state_dict(),
            'epoch': epochs,
            'test_metrics': test_metrics,
            'stage': 'stage2'
        }, save_path / 'best_model_stage2.pth')

        print(f"\n Stage2: {save_path / 'best_model_stage2.pth'}")

        return test_metrics

    def run_full_pipeline(self):

        results = {}

        results['stage1'] = self.stage1_source_pretrain(epochs=1000)
        results['stage2'] = self.stage2_target_finetune(epochs=1000)


        self._print_metrics("Stage1", results['stage1'])

        stage1_auc = results['stage1']['auc']
        stage2_auc = results['stage2']['auc']
        improvement = stage2_auc - stage1_auc

        print(f"\nStage 1: AUC = {stage1_auc:.4f}")
        print(f"Stage 2: AUC = {stage2_auc:.4f}")

        return results

    def _print_metrics(self, title, metrics):
        """格式化打印指标"""
        print(f"\n{title}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  F1 (Weighted): {metrics['f1']:.4f}")
        print(f"  Pos F1: {metrics['f1_pos']:.4f}")
        print(f"  Pos Precision: {metrics['precision_pos']:.4f}")
        print(f"  Pos Recall: {metrics['recall_pos']:.4f}")
        print(f"  H-measure: {metrics['h_measure']:.4f}")
        print(f"  Specificity (Type I): {metrics['specificity']:.4f}")
        print(f"  Sensitivity (Type II): {metrics['sensitivity']:.4f}")

    @torch.no_grad()
    def _forward_teacher(self, data):
        self.teacher_model.eval()
        return self.teacher_model(data.x_dict, data.edge_index_dict, return_features=True)