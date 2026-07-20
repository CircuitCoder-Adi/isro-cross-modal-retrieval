import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, sar_emb, optical_emb):

        logits = torch.matmul(
            sar_emb,
            optical_emb.T
        ) / self.temperature

        labels = torch.arange(
            sar_emb.size(0),
            device=sar_emb.device
        )

        loss_sar = F.cross_entropy(
            logits,
            labels
        )

        loss_optical = F.cross_entropy(
            logits.T,
            labels
        )

        loss = (
            loss_sar +
            loss_optical
        ) / 2

        return loss