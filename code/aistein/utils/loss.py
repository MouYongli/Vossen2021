import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, reduction='mean', ignore_index=-1, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction
        self.cuda = cuda

    def build_loss(self, mode='bce'):
        """Choices: ['bce', 'ce' or 'focal']"""
        if mode == 'bce':
            return self.BinaryEntropyLoss
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def BinaryEntropyLoss(self, logit, target):
        criterion = nn.BCELoss(weight=self.weight, reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(torch.squeeze(logit, dim=1), target.float())
        return loss

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()
        logpt = -criterion(logit, target)
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.randint(0, 2, (1, 7, 7)).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

    y = torch.rand(4, 1, 512, 512)
    t = torch.randint(0,1,(4,512,512))
    y = torch.squeeze(y, dim=1)
    criterion = nn.BCELoss()
    print(criterion(y, t.float()))
    print(loss.BinaryEntropyLoss(y, t).item())




