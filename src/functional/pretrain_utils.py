import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from algorithm.graph_augment import graph_views
from functional.common import parse_backbone_output


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, hidden_dim, temperature=0.5):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.temperature = temperature

    def forward(self, zi, zj):
        batch_size = zi.size(0)
        x1_abs = zi.norm(dim=1)
        x2_abs = zj.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        return -torch.log(loss).mean()


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, hidden_dim, feature_num):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, feature_num),
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input_features, hidden_features):
        reconstruction_features = self.decoder(hidden_features)
        return self.loss_fn(input_features, reconstruction_features)


def build_contrastive_loaders(data, batch_size):
    augs = random.choices(['dropN', 'permE', 'maskN'], k=2)
    aug_ratio = random.randint(1, 3) * 1.0 / 10

    view_list_1 = []
    view_list_2 = []

    for graph in data:
        view_g = graph_views(data=graph, aug=augs[0], aug_ratio=aug_ratio)
        view_list_1.append(Data(x=view_g.x, edge_index=view_g.edge_index))

        view_g = graph_views(data=graph, aug=augs[1], aug_ratio=aug_ratio)
        view_list_2.append(Data(x=view_g.x, edge_index=view_g.edge_index))

    loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False, num_workers=4)
    loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader1, loader2


def build_teacher(model, device):
    teacher = deepcopy(model).eval().to(device)
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def build_pretrain_optimizer(
    model,
    frontdoor,
    loss_fn,
    gco_model,
    reconstruct,
    data,
    learning_rate,
    weight_decay,
):
    rec_loss_fn = None
    params = list(model.parameters()) + list(frontdoor.parameters()) + list(loss_fn.parameters())

    if gco_model is not None:
        params = list(gco_model.parameters()) + params

    if reconstruct != 0.0:
        rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features)
        params += list(rec_loss_fn.parameters())

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, params),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer, rec_loss_fn


def edge_reconstruction_loss(h, edge_index, num_neg=512):
    src, dst = edge_index
    pos_score = torch.sigmoid((h[src] * h[dst]).sum(dim=-1))
    pos_loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))

    num_nodes = h.size(0)
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=h.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=h.device)
    neg_score = torch.sigmoid((h[neg_src] * h[neg_dst]).sum(dim=-1))
    neg_loss = F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
    return (pos_loss + neg_loss) / 2.0


def compute_node_task_losses(hi, hj, batch1, batch2):
    node_align_loss = F.mse_loss(hi.mean(0), hj.mean(0))
    lambda_node = 0.05

    edge_loss_i = edge_reconstruction_loss(hi, batch1.edge_index)
    edge_loss_j = edge_reconstruction_loss(hj, batch2.edge_index)
    lambda_edge = 0.1

    mean_align = F.mse_loss(hi.mean(0), hj.mean(0))
    std_align = F.mse_loss(hi.std(0), hj.std(0))
    lambda_dist = 0.05

    return (
        lambda_node * node_align_loss
        + lambda_edge * (edge_loss_i + edge_loss_j)
        + lambda_dist * (mean_align + std_align)
    )


def compute_frontdoor_losses(frontdoor, hi, hj):
    mediator_i = frontdoor.mediator(hi)
    mediator_j = frontdoor.mediator(hj)

    rec_loss_i = F.mse_loss(frontdoor.reconstructor(mediator_i), hi.detach())
    rec_loss_j = F.mse_loss(frontdoor.reconstructor(mediator_j), hj.detach())
    cons_loss = F.mse_loss(mediator_i.mean(0), mediator_j.mean(0))
    return rec_loss_i, rec_loss_j, cons_loss


def compute_cfd_loss(zi, zj, t_z1, lambda_causal=0.1):
    score = (zi * t_z1).sum(dim=1, keepdim=True)
    corrected = zi - lambda_causal * score * zi
    return F.mse_loss(zi, corrected) + F.mse_loss(zj, corrected)


def run_pretrain_epoch(
    epoch_idx,
    loaders,
    model,
    teacher,
    frontdoor,
    loss_fn,
    optimizer,
    loss_metric,
    device,
    task,
    lambda_causal,
    lambda_env,
    lambda_rec,
    lambda_cons,
    lambda_eps,
    lambda_cfd,
    reconstruct,
    gco_model=None,
    rec_loss_fn=None,
):
    pbar = tqdm(zip(*loaders), total=len(loaders[0]), ncols=100, desc=f'Epoch {epoch_idx}, Loss: inf')

    for batch1, batch2 in pbar:
        if gco_model is not None:
            batch1 = gco_model(batch1)
            batch2 = gco_model(batch2)

        optimizer.zero_grad()

        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        out1 = model(batch1)
        out2 = model(batch2)

        zi, hi, wi, reg1 = parse_backbone_output(out1)
        zj, hj, wj, reg2 = parse_backbone_output(out2)

        loss = loss_fn(zi, zj)

        if (wi is not None) and (wj is not None):
            env_loss = F.mse_loss(wi, wj)
            loss = loss + lambda_causal * lambda_env * env_loss

        if task == 'node':
            loss = loss + compute_node_task_losses(hi, hj, batch1, batch2)

        rec_loss_i, rec_loss_j, cons_loss = compute_frontdoor_losses(frontdoor, hi, hj)
        loss = loss + lambda_causal * lambda_rec * (rec_loss_i + rec_loss_j)
        loss = loss + lambda_causal * lambda_cons * cons_loss

        with torch.no_grad():
            t_out1 = teacher(batch1)
            t_out2 = teacher(batch2)
            t_z1, _, _, _ = parse_backbone_output(t_out1)
            t_z2, _, _, _ = parse_backbone_output(t_out2)

        cfd_loss = compute_cfd_loss(zi, zj, t_z1, device, lambda_causal, lambda_eps)
        loss = loss + lambda_causal * lambda_cfd * cfd_loss

        if reconstruct != 0.0 and hi is not None and hj is not None:
            loss = loss + reconstruct * (rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch2.x, hj))

        if (reg1 != 0.0) or (reg2 != 0.0):
            loss = loss + reg1 + reg2

        loss.backward()
        optimizer.step()

        loss_metric.update(loss.item(), batch1.size(0))
        pbar.set_description(f'Epoch {epoch_idx}, Loss {loss_metric.compute():.4f}', refresh=True)

    pbar.close()
