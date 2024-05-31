import torch


class GANBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input, target, disc_fake):
        return self.loss_fn(disc_fake, torch.ones_like(disc_fake))
