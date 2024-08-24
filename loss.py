from torch import nn
import torch

class Loss(nn.Module):
    def __init__(self, mode = 'vanilla', label_is_real = 1.0, label_is_fake = 0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(label_is_real))
        self.register_buffer('fake_label', torch.tensor(label_is_fake))

        if mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'L1':
            self.loss == nn.L1Loss()

    def __call__(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        labels = labels.expand_as(preds)

        loss = self.loss(preds, labels)
        return loss

