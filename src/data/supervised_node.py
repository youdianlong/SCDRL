import os

import torch


def get_supervised_node_data(dataset, ratios, seed=777, cache_dir='storage/.cache', few_shot=1, node_feature_dim=0):
    dataset_cache_root = cache_dir
    cache_dir = os.path.join(cache_dir, dataset)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'fewshot_node_{few_shot}_s{seed}.pt')

    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from .utils import iterate_datasets, preprocess

    data = preprocess(next(iterate_datasets(dataset, cache_dir=dataset_cache_root)), node_feature_dim=node_feature_dim)
    num_classes = torch.unique(data.y).size(0)
    torch.manual_seed(seed)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        idx = idx[torch.randperm(len(idx))]

        train_idx = idx[:few_shot]
        val_num = max(1, len(idx) // 10)
        val_idx = idx[few_shot:few_shot + val_num]
        test_idx = idx[few_shot + val_num:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    results = [{'train': [data], 'val': [data], 'test': [data]}, num_classes]
    torch.save(results, cache_path)
    return results
