from importlib import import_module
import torch


class Model(torch.nn.Module):
    def __init__(self, backbone, answering=torch.nn.Identity(), saliency=torch.nn.Identity()):
        super().__init__()
        self.backbone = backbone
        self.answering = answering
        self.saliency = saliency

    @staticmethod
    def _parse_backbone_output(backbone_out):
        reg = 0.0
        weights = None
        node_emb = None

        if isinstance(backbone_out, tuple):
            if len(backbone_out) == 2 and isinstance(backbone_out[0], tuple):
                nested_out, reg = backbone_out
                graph_emb = nested_out[0]
                if len(nested_out) > 1:
                    node_emb = nested_out[1]
                if len(nested_out) > 2:
                    weights = nested_out[2]
            elif len(backbone_out) == 3:
                graph_emb, node_emb, weights = backbone_out
            elif len(backbone_out) == 2 and hasattr(backbone_out[1], 'dim') and backbone_out[1].dim() == 2:
                graph_emb, node_emb = backbone_out
            else:
                graph_emb = backbone_out[0]
        else:
            graph_emb = backbone_out

        return graph_emb, node_emb, weights, reg

    def forward(self, data):
        data.x = self.saliency(data.x)
        backbone_out = self.backbone(data)
        graph_emb, node_emb, weights, reg = self._parse_backbone_output(backbone_out)

        if isinstance(self.answering, torch.nn.Identity):
            return (graph_emb, node_emb, weights), reg

        if (node_emb is not None) and hasattr(data, 'y') and data.y.size(0) == node_emb.size(0):
            out = self.answering(node_emb)
        else:
            out = self.answering(graph_emb)
        return out


def get_model(backbone_kwargs, answering_kwargs=None, saliency_kwargs=None):
    backbone = import_module(f"model.backbone.{backbone_kwargs.pop('name')}").get_model(**backbone_kwargs)
    answering = torch.nn.Identity() if answering_kwargs is None else import_module(
        f"model.answering.{answering_kwargs.pop('name')}"
    ).get_model(**answering_kwargs)
    saliency = torch.nn.Identity() if saliency_kwargs is None else import_module(
        f"model.saliency.{saliency_kwargs.pop('name')}"
    ).get_model(**saliency_kwargs)
    return Model(backbone, answering, saliency)
