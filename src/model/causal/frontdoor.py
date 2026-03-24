import torch
import torch.nn as nn
import torch.nn.functional as F


class FrontdoorMediatorLite(nn.Module):
    def __init__(self, in_dim, hid_dim, num_class, dropout=0.2, lambda_rec=0.1):
        super().__init__()
        self.lambda_rec = lambda_rec

        # simpler mediator
        self.mediator = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # simple reconstruction branch
        self.reconstructor = nn.Linear(hid_dim, in_dim)

        # plain classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_dim + hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_class),
        )

    def forward(self, h_x):
        # mediator feature
        m = self.mediator(h_x)

        # reconstruction loss
        x_rec = self.reconstructor(m)
        rec_loss = F.mse_loss(x_rec, h_x.detach())

        feat = torch.cat([h_x, m], dim=-1)
        logits = self.classifier(feat)
        log_prob = F.log_softmax(logits, dim=-1)

        reg_loss = self.lambda_rec * rec_loss
        return log_prob, m, reg_loss


def get_model(num_class, hid_dim=128, dropout=0.2, lambda_rec=0.1):
    return FrontdoorMediatorLite(
        in_dim=hid_dim,
        hid_dim=hid_dim,
        num_class=num_class,
        dropout=dropout,
        lambda_rec=lambda_rec,
    )