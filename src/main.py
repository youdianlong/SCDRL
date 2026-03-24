import argparse
import json
import os
import random
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATASETS = [
    'wisconsin', 'texas', 'cornell', 'chameleon', 'squirrel',
    'cora', 'citeseer', 'pubmed', 'computers', 'photo',
]


def _load_config_file(config_file):
    if not config_file:
        return {}
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError('Config file must contain a flat JSON object.')
    return config


def _cfg(config, key, default):
    return config.get(key, default)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def build_parser(config=None):
    config = config or {}
    parser = argparse.ArgumentParser('UCDRL')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    pretrain = subparsers.add_parser('pretrain', help='Run multi-source pretraining')
    pretrain.add_argument('--config_file', type=str, default=None, help='optional flat JSON config file')
    pretrain.add_argument('--source_datasets', nargs='+', type=str, default=_cfg(config, 'source_datasets', None), choices=DATASETS, required=_cfg(config, 'source_datasets', None) is None, help='source datasets for multi-source pretraining')
    pretrain.add_argument('--task', type=str, default=_cfg(config, 'task', 'node'), choices=['node', 'graph'], help='downstream task type')
    pretrain.add_argument('--cache_dir', type=str, default=_cfg(config, 'cache_dir', 'storage/.cache'))
    pretrain.add_argument('--save_dir', type=str, default=_cfg(config, 'save_dir', 'storage/UCDRL/pretrained_model'))
    pretrain.add_argument('--seed', type=int, default=_cfg(config, 'seed', 777))
    pretrain.add_argument('--node_feature_dim', type=int, default=_cfg(config, 'node_feature_dim', 0))
    pretrain.add_argument('--reconstruct', type=float, default=_cfg(config, 'reconstruct', 0.0))
    pretrain.add_argument('--backbone', type=str, default=_cfg(config, 'backbone', 'MoE'), choices=['MoE'])
    pretrain.add_argument('--hid_dim', type=int, default=_cfg(config, 'hid_dim', 128))
    pretrain.add_argument('--moe_num_conv_layers', type=int, default=_cfg(config, 'moe_num_conv_layers', 2))
    pretrain.add_argument('--moe_dropout', type=float, default=_cfg(config, 'moe_dropout', 0.2))
    pretrain.add_argument('--moe_epsilon', type=float, default=_cfg(config, 'moe_epsilon', 0.1))
    pretrain.add_argument('--moe_num_experts', type=int, default=_cfg(config, 'moe_num_experts', 10))
    pretrain.add_argument('--moe_tau', type=float, default=_cfg(config, 'moe_tau', 1.0))
    pretrain.add_argument('--moe_reg_lambda', type=float, default=_cfg(config, 'moe_reg_lambda', 0.01))
    pretrain.add_argument('--saliency_model', type=str, default=_cfg(config, 'saliency_model', 'none'), choices=['none', 'mlp'])
    pretrain.add_argument('--saliency_hid_dim', type=int, default=_cfg(config, 'saliency_hid_dim', 4096))
    pretrain.add_argument('--saliency_num_layers', type=int, default=_cfg(config, 'saliency_num_layers', 2))
    pretrain.add_argument('--pretrain_method', type=str, default=_cfg(config, 'pretrain_method', 'ucdrl_pretrain'), help='pretraining objective')
    pretrain.add_argument('--lr', type=float, default=_cfg(config, 'lr', 1e-2), help='learning rate')
    pretrain.add_argument('--weight_decay', type=float, default=_cfg(config, 'weight_decay', 1e-5))
    pretrain.add_argument('--epochs', type=int, default=_cfg(config, 'epochs', 100))
    pretrain.add_argument('--batch_size', type=int, default=_cfg(config, 'batch_size', 10))
    pretrain.add_argument('--noise_switch', type=_str2bool, default=_cfg(config, 'noise_switch', False))
    pretrain.add_argument('--cross_link', type=int, default=_cfg(config, 'cross_link', 0))
    pretrain.add_argument('--cross_link_ablation', type=_str2bool, default=_cfg(config, 'cross_link_ablation', False))
    pretrain.add_argument('--dynamic_edge', type=str, default=_cfg(config, 'dynamic_edge', 'none'), choices=['internal', 'external', 'internal_external', 'similarity', 'none'])
    pretrain.add_argument('--dynamic_prune', type=float, default=_cfg(config, 'dynamic_prune', 0.1))
    pretrain.add_argument('--cl_init_method', type=str, default=_cfg(config, 'cl_init_method', 'learnable'), choices=['mean', 'sum', 'learnable', 'simple', 'none'])
    pretrain.add_argument('--split_method', type=str, default=_cfg(config, 'split_method', 'RandomWalk'), choices=['metis', 'RandomWalk'])
    pretrain.add_argument('--lambda_causal', type=float, default=_cfg(config, 'lambda_causal', 1.0))
    pretrain.add_argument('--lambda_env', type=float, default=_cfg(config, 'lambda_env', 0.5))
    pretrain.add_argument('--lambda_rec', type=float, default=_cfg(config, 'lambda_rec', 0.1))
    pretrain.add_argument('--lambda_cons', type=float, default=_cfg(config, 'lambda_cons', 0.05))
    pretrain.add_argument('--lambda_eps', type=float, default=_cfg(config, 'lambda_eps', 0.01))
    pretrain.add_argument('--lambda_cfd', type=float, default=_cfg(config, 'lambda_cfd', 0.2))

    adapt = subparsers.add_parser('adapt', help='Run downstream adaptation / finetuning')
    adapt.add_argument('--config_file', type=str, default=None, help='optional flat JSON config file')
    adapt.add_argument('--target_dataset', type=str, default=_cfg(config, 'target_dataset', None), choices=DATASETS, required=_cfg(config, 'target_dataset', None) is None, help='target dataset for downstream evaluation')
    adapt.add_argument('--task', type=str, default=_cfg(config, 'task', 'node'), choices=['node', 'graph'])
    adapt.add_argument('--cache_dir', type=str, default=_cfg(config, 'cache_dir', 'storage/.cache'))
    adapt.add_argument('--save_dir', type=str, default=_cfg(config, 'save_dir', 'storage/UCDRL/result'))
    adapt.add_argument('--seed', type=int, default=_cfg(config, 'seed', 777))
    adapt.add_argument('--few_shot', type=int, default=_cfg(config, 'few_shot', 1), help='few-shot samples per class')
    adapt.add_argument('--node_feature_dim', type=int, default=_cfg(config, 'node_feature_dim', 0))
    adapt.add_argument('--ratios', nargs=3, type=float, default=_cfg(config, 'ratios', [0.1, 0.1, 0.8]), metavar=('TRAIN', 'VAL', 'TEST'))
    adapt.add_argument('--reconstruct', type=float, default=_cfg(config, 'reconstruct', 0.0))
    adapt.add_argument('--backbone', type=str, default=_cfg(config, 'backbone', 'MoE'), choices=['MoE'])
    adapt.add_argument('--hid_dim', type=int, default=_cfg(config, 'hid_dim', 128))
    adapt.add_argument('--moe_num_conv_layers', type=int, default=_cfg(config, 'moe_num_conv_layers', 2))
    adapt.add_argument('--moe_dropout', type=float, default=_cfg(config, 'moe_dropout', 0.2))
    adapt.add_argument('--moe_epsilon', type=float, default=_cfg(config, 'moe_epsilon', 0.1))
    adapt.add_argument('--moe_num_experts', type=int, default=_cfg(config, 'moe_num_experts', 10))
    adapt.add_argument('--moe_tau', type=float, default=_cfg(config, 'moe_tau', 1.0))
    adapt.add_argument('--moe_reg_lambda', type=float, default=_cfg(config, 'moe_reg_lambda', 0.01))
    adapt.add_argument('--saliency_model', type=str, default=_cfg(config, 'saliency_model', 'none'), choices=['none', 'mlp'])
    adapt.add_argument('--saliency_hid_dim', type=int, default=_cfg(config, 'saliency_hid_dim', 4096))
    adapt.add_argument('--saliency_num_layers', type=int, default=_cfg(config, 'saliency_num_layers', 2))
    adapt.add_argument('--answering_model', type=str, default=_cfg(config, 'answering_model', 'mlp'), choices=['mlp'])
    adapt.add_argument('--answering_num_layers', type=int, default=_cfg(config, 'answering_num_layers', 2))
    adapt.add_argument('--adapt_method', type=str, default=_cfg(config, 'adapt_method', 'finetune'), choices=['finetune'])
    adapt.add_argument('--pretrained_file', type=str, default=_cfg(config, 'pretrained_file', None), required=_cfg(config, 'pretrained_file', None) is None)
    adapt.add_argument('--repeat_times', type=int, default=_cfg(config, 'repeat_times', 5))
    adapt.add_argument('--epochs', type=int, default=_cfg(config, 'epochs', 100))
    adapt.add_argument('--batch_size', type=int, default=_cfg(config, 'batch_size', 10))
    adapt.add_argument('--lr', type=float, default=_cfg(config, 'lr', 1e-4), help='learning rate')
    adapt.add_argument('--weight_decay', type=float, default=_cfg(config, 'weight_decay', 1e-5))
    adapt.add_argument('--backbone_tuning', type=_str2bool, default=_cfg(config, 'backbone_tuning', False))
    adapt.add_argument('--saliency_tuning', type=_str2bool, default=_cfg(config, 'saliency_tuning', False))
    adapt.add_argument('--lambda_rec', type=float, default=_cfg(config, 'lambda_rec', 0.1))
    adapt.add_argument('--lambda_cons', type=float, default=_cfg(config, 'lambda_cons', 0.05))

    return parser


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('mode', choices=['pretrain', 'adapt'])
    pre_parser.add_argument('--config_file', type=str, default=None)
    known, _ = pre_parser.parse_known_args(argv)
    config = _load_config_file(known.config_file)
    parser = build_parser(config=config)
    return parser.parse_args(argv)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def dump_args(args):
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'config.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def main(argv=None):
    args = parse_args(argv)
    set_random_seed(args.seed)
    dump_args(args)
    import_module(f'functional.{args.mode}').run(args)


if __name__ == '__main__':
    main()
