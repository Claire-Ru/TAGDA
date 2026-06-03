
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)



class DomainDiscriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def compute_a_distance(domain_disc, h_src, h_tgt, device):

    domain_disc.eval()
    with torch.no_grad():
        pred_src = domain_disc(h_src)
        pred_src = (pred_src > 0.5).float()

        pred_tgt = domain_disc(h_tgt)
        pred_tgt = (pred_tgt > 0.5).float()

        error_src = (pred_src == 0).float().mean()
        error_tgt = (pred_tgt == 1).float().mean()
        error_rate = (error_src + error_tgt) / 2.0

        a_dist = 2.0 * (1.0 - 2.0 * error_rate)
        a_dist = torch.clamp(a_dist, min=0.0, max=2.0)

    domain_disc.train()
    return a_dist.item()



def mmd_loss(X, Y, bandwidth_list=None, max_mmd_samples=2048):

    if bandwidth_list is None:
        bandwidth_list = [1.0, 10.0, 100.0]

    if X.size(0) > max_mmd_samples:
        indices_x = torch.randperm(X.size(0), device=X.device)[:max_mmd_samples]
        X = X[indices_x]
    if Y.size(0) > max_mmd_samples:
        indices_y = torch.randperm(Y.size(0), device=Y.device)[:max_mmd_samples]
        Y = Y[indices_y]

    n_x, n_y = X.size(0), Y.size(0)
    if n_x <= 1 or n_y <= 1:
        return torch.tensor(0.0, device=X.device)

    XX = X.unsqueeze(1) - X.unsqueeze(0)
    XX_dist = (XX ** 2).sum(2)

    YY = Y.unsqueeze(1) - Y.unsqueeze(0)
    YY_dist = (YY ** 2).sum(2)

    XY = X.unsqueeze(1) - Y.unsqueeze(0)
    XY_dist = (XY ** 2).sum(2)

    XX_kernel = sum([torch.exp(-XX_dist / bw) for bw in bandwidth_list])
    YY_kernel = sum([torch.exp(-YY_dist / bw) for bw in bandwidth_list])
    XY_kernel = sum([torch.exp(-XY_dist / bw) for bw in bandwidth_list])

    mmd = XX_kernel.mean() + YY_kernel.mean() - 2 * XY_kernel.mean()
    return torch.relu(mmd)


def coral_loss(h_s, h_t):
    n_s, d = h_s.size()
    n_t, _ = h_t.size()

    if n_s <= 1 or n_t <= 1:
        return torch.tensor(0.0, device=h_s.device)

    h_s_centered = h_s - h_s.mean(dim=0, keepdim=True)
    h_t_centered = h_t - h_t.mean(dim=0, keepdim=True)

    C_s = (h_s_centered.t() @ h_s_centered) / (n_s - 1)
    C_t = (h_t_centered.t() @ h_t_centered) / (n_t - 1)

    loss = torch.norm(C_s - C_t, p='fro') ** 2
    return loss



class ClassConditionalAlignment(nn.Module):

    def __init__(self, num_classes=2, use_mmd=True,
                 shared_dim=48, private_dim=16):
        super().__init__()
        self.num_classes = num_classes
        self.use_mmd = use_mmd

        self.grl = GradientReversalLayer(alpha=1.0)

        self.disc_shared = DomainDiscriminator(shared_dim, hidden_dim=128)
        self.disc_private = DomainDiscriminator(private_dim, hidden_dim=64)

        self.a_dist_shared = 0.0
        self.a_dist_private = 0.0

    def update_grl_alpha(self, progress):

        alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0
        self.grl.alpha = alpha.item()

    def compute_domain_loss_single(self, discriminator, h_src, h_tgt):

        if h_src.size(0) == 0 or h_tgt.size(0) == 0:
            return torch.tensor(0.0, device=h_src.device)
        if h_src.size(0) < 2 or h_tgt.size(0) < 2:
            return torch.tensor(0.0, device=h_src.device)

        h_src_grl = self.grl(h_src)
        h_tgt_grl = self.grl(h_tgt)

        pred_src = discriminator(h_src_grl)
        pred_tgt = discriminator(h_tgt_grl)

        loss_src = F.binary_cross_entropy(pred_src, torch.ones_like(pred_src))
        loss_tgt = F.binary_cross_entropy(pred_tgt, torch.zeros_like(pred_tgt))

        return (loss_src + loss_tgt) / 2.0

    def compute_dynamic_adversarial_factor(self):

        total = 3 * self.a_dist_shared + self.a_dist_private
        if total < 1e-6:
            return 0.5
        omega = (3 * self.a_dist_shared) / total
        return omega

    def update_a_distances(self, h_shared_src, h_shared_tgt,
                           h_private_src, h_private_tgt):

        device = h_shared_src.device

        if h_shared_src.size(0) > 0 and h_shared_tgt.size(0) > 0:
            self.a_dist_shared = compute_a_distance(
                self.disc_shared, h_shared_src, h_shared_tgt, device
            )

        if h_private_src.size(0) > 0 and h_private_tgt.size(0) > 0:
            self.a_dist_private = compute_a_distance(
                self.disc_private, h_private_src, h_private_tgt, device
            )

    def forward(self, h_shared_src, labels_src, h_shared_tgt, labels_tgt,
                h_private_src=None, h_private_tgt=None,
                progress=0.0):

        loss_dict = {}
        alignment_loss = torch.tensor(0.0, device=h_shared_src.device)

        self.update_grl_alpha(progress)
        loss_dict['alpha'] = self.grl.alpha

        coral_loss_val = coral_loss(h_shared_src, h_shared_tgt)
        loss_dict['coral_loss'] = coral_loss_val.item()
        alignment_loss += coral_loss_val

        domain_loss_shared = self.compute_domain_loss_single(
            self.disc_shared, h_shared_src, h_shared_tgt
        )
        loss_dict['domain_loss_shared'] = domain_loss_shared.item()

        domain_loss_private = torch.tensor(0.0, device=h_shared_src.device)
        if h_private_src is not None and h_private_tgt is not None:
            domain_loss_private = self.compute_domain_loss_single(
                self.disc_private, h_private_src, h_private_tgt
            )
            loss_dict['domain_loss_private'] = domain_loss_private.item()

        if h_private_src is not None and h_private_tgt is not None:
            self.update_a_distances(
                h_shared_src, h_shared_tgt,
                h_private_src, h_private_tgt
            )

        loss_dict['a_dist_shared'] = self.a_dist_shared
        loss_dict['a_dist_private'] = self.a_dist_private

        omega = self.compute_dynamic_adversarial_factor()
        loss_dict['omega'] = omega

        total_domain_loss = (
            omega * domain_loss_shared +
            (1 - omega) * domain_loss_private
        )
        loss_dict['total_domain_loss'] = total_domain_loss.item()

        alignment_loss += total_domain_loss
        loss_dict['total_align_loss'] = alignment_loss.item()

        return alignment_loss, loss_dict