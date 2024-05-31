import torch


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input, target, disc_real, disc_fake):
        loss_real = self.loss_fn(disc_real, torch.ones_like(disc_real))
        loss_fake = self.loss_fn(disc_fake, torch.zeros_like(disc_fake))
        return loss_real + loss_fake
