import os

import torch
from torchmetrics import MeanMetric

from functional.pretrain_utils import (
    ContrastiveLoss,
    build_contrastive_loaders,
    build_pretrain_optimizer,
    build_teacher,
    run_pretrain_epoch,
)


def run(args):
    from data import get_clustered_data
    from model import get_model

    dataset = args.source_datasets
    gco_model, raw_data = None, None

    if args.saliency_model == 'mlp':
        data = get_clustered_data(
            dataset,
            cache_dir=args.cache_dir,
            cross_link=args.cross_link,
            cl_init_method=args.cl_init_method,
            cross_link_ablation=args.cross_link_ablation,
            dynamic_edge=args.dynamic_edge,
            dynamic_prune=args.dynamic_prune,
            split_method=args.split_method,
            node_feature_dim=args.node_feature_dim,
        )
    else:
        with torch.no_grad():
            data, gco_model, raw_data = get_clustered_data(
                dataset,
                cache_dir=args.cache_dir,
                cross_link=args.cross_link,
                cl_init_method=args.cl_init_method,
                cross_link_ablation=args.cross_link_ablation,
                dynamic_edge=args.dynamic_edge,
                dynamic_prune=args.dynamic_prune,
                split_method=args.split_method,
                node_feature_dim=args.node_feature_dim,
            )

    if isinstance(data, tuple):
        data, gco_model, raw_data = data

    model = get_model(
        backbone_kwargs={
            'name': args.backbone,
            'num_features': data[0].x.size(-1),
            'hid_dim': args.hid_dim,
            'num_conv_layers': args.moe_num_conv_layers,
            'dropout': args.moe_dropout,
            'epsilon': args.moe_epsilon,
            'num_experts': args.moe_num_experts,
            'tau': args.moe_tau,
            'reg_lambda': args.moe_reg_lambda,
            'reconstruct': args.reconstruct,
        },
        saliency_kwargs={
            'name': args.saliency_model,
            'feature_dim': data[0].x.size(-1),
            'hid_dim': args.saliency_hid_dim,
            'num_layers': args.saliency_num_layers,
        } if args.saliency_model != 'none' else None,
    )

    if args.pretrain_method in {'ucdrl_pretrain'}:
        model = graph_cl_pretrain(data, model, gco_model, raw_data, args)
    else:
        raise NotImplementedError(f'Unknown method: {args.pretrain_method}')

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, ','.join(dataset) + '_pretrained_model.pt'))


def graph_cl_pretrain(data, model, gco_model, raw_data, args):
    from copy import deepcopy

    from data.contrastive import update_graph_list_param
    from model.causal.frontdoor import FrontdoorMediator

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    frontdoor = FrontdoorMediator(
        in_dim=model.backbone.hidden_dim,
        hid_dim=model.backbone.hidden_dim,
        num_class=0,
        dropout=0.2,
        lambda_rec=args.lambda_rec,
        lambda_cons=args.lambda_cons,
    ).to(device)

    teacher = build_teacher(model, device)
    loss_fn = ContrastiveLoss(model.backbone.hidden_dim).to(device)
    loss_fn.train()
    model.train()

    optimizer, rec_loss_fn = build_pretrain_optimizer(
        model=model,
        frontdoor=frontdoor,
        loss_fn=loss_fn,
        gco_model=gco_model,
        reconstruct=args.reconstruct,
        data=data,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    if rec_loss_fn is not None:
        rec_loss_fn = rec_loss_fn.to(device)
        rec_loss_fn.train()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_metric = MeanMetric()

    best_loss = float('inf')
    best_model = None
    last_updated_data = data

    for e in range(args.epochs):
        loss_metric.reset()

        if args.cross_link > 0 and args.cl_init_method == 'learnable':
            if args.split_method == 'RandomWalk':
                last_updated_data = deepcopy(data)
            loaders = build_contrastive_loaders(data, args.batch_size)
        elif e == 0:
            loaders = build_contrastive_loaders(data, args.batch_size)

        run_pretrain_epoch(
            epoch_idx=e,
            loaders=loaders,
            model=model,
            teacher=teacher,
            frontdoor=frontdoor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            loss_metric=loss_metric,
            device=device,
            task=args.task,
            lambda_causal=args.lambda_causal,
            lambda_env=args.lambda_env,
            lambda_rec=args.lambda_rec,
            lambda_cons=args.lambda_cons,
            lambda_eps=args.lambda_eps,
            lambda_cfd=args.lambda_cfd,
            reconstruct=args.reconstruct,
            gco_model=gco_model,
            rec_loss_fn=rec_loss_fn,
        )

        if gco_model is not None:
            data = update_graph_list_param(last_updated_data, gco_model)
            gco_model.update_last_params()

        current_loss = loss_metric.compute()
        if current_loss < best_loss:
            best_loss = current_loss
            best_model = deepcopy(model)

        lr_scheduler.step()

    return best_model
