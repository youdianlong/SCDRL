import os
from copy import deepcopy

import numpy as np
import torch

from functional.adapt_utils import build_metrics, run_eval_epoch, run_train_epoch
from model.causal.frontdoor import FrontdoorMediator


def _set_random_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _build_model(args, datasets, num_classes):
    from model import get_model

    return get_model(
        backbone_kwargs={
            'name': args.backbone,
            'num_features': datasets['train'][0].x.size(-1),
            'hid_dim': args.hid_dim,
            'num_conv_layers': args.moe_num_conv_layers,
            'dropout': args.moe_dropout,
            'epsilon': args.moe_epsilon,
            'num_experts': args.moe_num_experts,
            'tau': args.moe_tau,
            'reg_lambda': args.moe_reg_lambda,
            'reconstruct': args.reconstruct,
        },
        answering_kwargs={
            'name': args.answering_model,
            'num_class': num_classes,
            'hid_dim': args.hid_dim,
            'num_layers': args.answering_num_layers,
        },
        saliency_kwargs={
            'name': args.saliency_model,
            'feature_dim': datasets['train'][0].x.size(-1),
            'hid_dim': args.saliency_hid_dim,
            'num_layers': args.saliency_num_layers,
        } if args.saliency_model != 'none' else None,
    )


def run(args):
    from torch_geometric.loader import DataLoader

    if args.task == 'graph':
        from data import get_supervised_data
        datasets, num_classes = get_supervised_data(
            args.target_dataset,
            ratios=args.ratios,
            seed=args.seed,
            cache_dir=args.cache_dir,
            few_shot=args.few_shot,
            node_feature_dim=args.node_feature_dim,
        )
    else:
        from data import get_supervised_node_data
        datasets, num_classes = get_supervised_node_data(
            args.target_dataset,
            ratios=args.ratios,
            seed=args.seed,
            cache_dir=args.cache_dir,
            few_shot=args.few_shot,
            node_feature_dim=args.node_feature_dim,
        )

    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=True, num_workers=4) for k, v in datasets.items()}
    pretrained_state = torch.load(args.pretrained_file, map_location='cpu')

    all_results = []
    for repeat_idx in range(args.repeat_times):
        _set_random_seed(args.seed + repeat_idx)
        model = _build_model(args, datasets, num_classes)
        model.load_state_dict(pretrained_state, strict=False)

        if args.adapt_method == 'finetune':
            results = finetune(loaders, model, args)
        else:
            raise NotImplementedError(f'Unknown method: {args.adapt_method}')

        results.pop('model')
        all_results.append(results)

    for key in all_results[0].keys():
        print(f'{key}: {np.mean([r[key] for r in all_results]):.4f} ± {np.std([r[key] for r in all_results]):.4f}')

    os.makedirs(args.save_dir, exist_ok=True)
    result_path = os.path.join(args.save_dir, args.target_dataset + '_results.txt')
    with open(result_path, 'a+', encoding='utf-8') as f:
        f.write('-------------------------------------------------\n')
        for key in all_results[0].keys():
            message = (
                args.adapt_method
                + f' FT on All, Target Dataset: {args.target_dataset}, {key}: '
                + f'{np.mean([r[key] for r in all_results]):.4f} ± {np.std([r[key] for r in all_results]):.4f}\n'
            )
            f.write(message)


def finetune(loaders, model, args):
    model.backbone.requires_grad_(args.backbone_tuning)
    model.saliency.requires_grad_(args.saliency_tuning)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    frontdoor = FrontdoorMediator(
        in_dim=model.backbone.hidden_dim,
        hid_dim=model.backbone.hidden_dim,
        num_class=model.answering.num_class,
        dropout=0.2,
        lambda_rec=args.lambda_rec,
        lambda_cons=args.lambda_cons,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters()) + list(frontdoor.parameters())),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics = build_metrics(model.answering.num_class, device)
    best_acc = 0.0
    best_model = None
    best_frontdoor = None

    for e in range(args.epochs):
        run_train_epoch(e, loaders, model, frontdoor, optimizer, metrics, device)
        run_eval_epoch(e, loaders['val'], 'val', model, frontdoor, metrics, device)

        current_acc = metrics['acc'].compute()
        if current_acc > best_acc:
            best_acc = current_acc
            best_model = deepcopy(model)
            best_frontdoor = deepcopy(frontdoor)

    model = best_model if best_model is not None else model
    frontdoor = best_frontdoor if best_frontdoor is not None else frontdoor

    run_eval_epoch(args.epochs, loaders['test'], 'test', model, frontdoor, metrics, device)
    return {
        'acc': metrics['acc'].compute().item(),
        'auroc': metrics['auroc'].compute().item(),
        'f1': metrics['f1'].compute().item(),
        'model': model.state_dict(),
    }
