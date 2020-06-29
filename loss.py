import torch.nn as nn

class RPNloss(nn.Module):
    def __init__(self, lambd):
        super(RPNloss, self).__init__()
        self.lambd = lambd
        self.bceloss = nn.BCELoss()
        self.regloss = nn.SmoothL1Loss()

    def __call__(self, y, gt):
        return self.bceloss(y, gt)/8 + self.lambd*self.regloss(y, gt)/676
