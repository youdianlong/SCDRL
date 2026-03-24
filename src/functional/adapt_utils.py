import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score, MeanMetric
from tqdm import tqdm

from functional.common import parse_backbone_output, select_task_representation


def build_metrics(num_classes, device):
    return {
        'loss': MeanMetric().to(device),
        'acc': Accuracy(task='multiclass', num_classes=num_classes).to(device),
        'f1': F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'auroc': AUROC(task='multiclass', num_classes=num_classes).to(device),
    }


def reset_metrics(metrics, keys=None):
    selected_keys = keys if keys is not None else metrics.keys()
    for key in selected_keys:
        metrics[key].reset()


def forward_frontdoor(model, frontdoor, batch, split, device):
    batch = batch.to(device)
    backbone_out = model.backbone(batch)
    graph_emb, node_emb, _, _ = parse_backbone_output(backbone_out)
    h_x, target, mask = select_task_representation(batch, graph_emb, node_emb, split=split)
    log_prob, _, reg_loss = frontdoor(h_x)
    return batch, target, mask, log_prob, reg_loss


def update_eval_metrics(metrics, prob, pred, target, mask):
    if mask is not None:
        metrics['acc'].update(pred[mask], target[mask])
        metrics['f1'].update(pred[mask], target[mask])
        metrics['auroc'].update(prob[mask], target[mask])
    else:
        metrics['acc'].update(pred, target)
        metrics['f1'].update(pred, target)
        metrics['auroc'].update(prob, target)


def run_train_epoch(epoch_idx, loaders, model, frontdoor, optimizer, metrics, device):
    model.train()
    frontdoor.train()
    reset_metrics(metrics, keys=['loss'])

    pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {epoch_idx} Training, Loss: inf')
    for batch in pbar:
        optimizer.zero_grad()
        batch, target, mask, log_prob, reg_loss = forward_frontdoor(model, frontdoor, batch, 'train', device)

        if mask is not None:
            loss_cls = F.nll_loss(log_prob[mask], target[mask])
        else:
            loss_cls = F.nll_loss(log_prob, target)

        loss = loss_cls + reg_loss
        loss.backward()
        optimizer.step()

        metrics['loss'].update(loss.detach(), batch.size(0))
        pbar.set_description(f'Epoch {epoch_idx} Training Loss: {metrics["loss"].compute():.4f}', refresh=True)

    pbar.close()


def run_eval_epoch(epoch_idx, loader, split, model, frontdoor, metrics, device):
    model.eval()
    frontdoor.eval()
    reset_metrics(metrics, keys=['acc', 'f1', 'auroc'])

    prefix = 'Testing' if split == 'test' else f'Epoch {epoch_idx} Validation'
    pbar = tqdm(loader, total=len(loader), ncols=100, desc=f'{prefix}, Acc: 0., F1: 0.')

    with torch.no_grad():
        for batch in pbar:
            _, target, mask, log_prob, _ = forward_frontdoor(model, frontdoor, batch, split, device)
            prob = torch.exp(log_prob)
            pred = prob.argmax(dim=-1)
            update_eval_metrics(metrics, prob, pred, target, mask)
            pbar.set_description(
                f'{prefix} Acc: {metrics["acc"].compute():.4f}, '
                f'AUROC: {metrics["auroc"].compute():.4f}, '
                f'F1: {metrics["f1"].compute():.4f}',
                refresh=True,
            )

    pbar.close()
