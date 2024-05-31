import torch


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake):
        return self.loss_fn(
            pred_real, torch.ones_like(pred_real)
        ) + self.loss_fn(pred_fake, torch.zeros_like(pred_fake))
