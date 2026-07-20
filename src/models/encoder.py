import torch
import torch.nn as nn
from torchvision.models import resnet50

class Encoder(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        backbone = resnet50(weights="DEFAULT")

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.projector = nn.Linear(
            backbone.fc.in_features,
            embedding_dim
        )

    def forward(self, x):

        x = self.features(x)
        x = torch.flatten(x, 1)

        x = self.projector(x)

        x = nn.functional.normalize(x, dim=1)

        return x


class CrossModalModel(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.sar_encoder = Encoder(embedding_dim)
        self.optical_encoder = Encoder(embedding_dim)

    def forward(self, sar, optical):

        sar_emb = self.sar_encoder(sar)
        optical_emb = self.optical_encoder(optical)

        return sar_emb, optical_emb