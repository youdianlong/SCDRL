import torch

def parse_backbone_output(out):
    reg = 0.0
    weights = None
    node_emb = None

    if isinstance(out, tuple):
        if len(out) == 2 and isinstance(out[0], tuple):
            nested_out, reg = out
            graph_emb = nested_out[0]
            if len(nested_out) > 1:
                node_emb = nested_out[1]
            if len(nested_out) > 2:
                weights = nested_out[2]
        elif len(out) == 3:
            graph_emb, node_emb, weights = out
        elif len(out) == 2 and hasattr(out[1], 'dim') and out[1].dim() == 2:
            graph_emb, node_emb = out
        else:
            graph_emb = out[0]
    else:
        graph_emb = out

    return graph_emb, node_emb, weights, reg


def select_task_representation(batch, graph_emb, node_emb, split=None):
    if split is not None:
        mask_name = f'{split}_mask'
        if node_emb is not None and hasattr(batch, mask_name):
            return node_emb, batch.y, getattr(batch, mask_name)

    return graph_emb, batch.y, None
