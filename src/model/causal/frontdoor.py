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
        M = self.norm(self.mediator(h_x))

        x_rec = self.reconstructor(M)
        rec_loss = F.mse_loss(x_rec, h_x.detach())

        cons_loss = torch.mean((M.mean(0) - h_x.mean(0)) ** 2)

        total_reg = self.lambda_rec * rec_loss + self.lambda_cons * cons_loss

        if self.fusion is None:
            return None, M, total_reg

        B = h_x.size(0)
        if B <= 1:
            logits = self.fusion(torch.cat([h_x, M], dim=-1))
            log_prob = F.log_softmax(logits, dim=-1)
            return log_prob, M, total_reg

        n = min(self.n_tprime, B - 1)  

        base = torch.arange(B, device=h_x.device).unsqueeze(1)  # [B,1]
        offsets = torch.randint(1, B, (B, n), device=h_x.device)  # in [1, B-1]
        idx = (base + offsets) % B

        t_prime = h_x[idx]

        M_exp = M.unsqueeze(1).expand(B, n, M.size(-1))

        inp = torch.cat([t_prime, M_exp], dim=-1).reshape(B * n, -1)
        logits_samples = self.fusion(inp).reshape(B, n, self.num_class)  # [B, n, C]

        probs = F.softmax(logits_samples, dim=-1)  # [B, n, C]
        mean_probs = probs.mean(dim=1)  # [B, C]
        log_prob = torch.log(mean_probs.clamp_min(self.eps))  # [B, C]

        return log_prob, M, total_reg


def get_model(num_class, hid_dim=128, dropout=0.2, lambda_rec=0.1):
    return FrontdoorMediatorLite(
        in_dim=hid_dim,
        hid_dim=hid_dim,
        num_class=num_class,
        dropout=dropout,
        lambda_rec=lambda_rec,
    )
