import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
import torch.nn as nn
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.num_classes = num_classes
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder_s1 = build_decoder(1, backbone, BatchNorm)
        self.decoder_s2 = build_decoder(num_classes, backbone, BatchNorm)
        self.sigmoid = nn.Sigmoid()
        self.last_conv = nn.Sequential(nn.Conv2d(num_classes, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x1 = self.sigmoid(self.decoder_s1(x, low_level_feat))
        x2 = self.decoder_s2(x, low_level_feat)
        x = self.last_conv(x2 + torch.repeat_interleave(x1, self.num_classes, dim=1))
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.training:
            return x1, x
        else:
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder_s1, self.decoder_s2]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16, num_classes=8)
    model.eval()
    x = torch.rand(4, 3, 512, 512)
    y1, y2 = model(x)
    t2 = torch.randint(low=0, high=7, size=(4, 512, 512))
    ones = torch.ones((4, 512, 512))
    zeros = torch.zeros((4, 512, 512))
    t1 = torch.where(t2 == 1, zeros, ones)
    print(t1)
    bce = nn.BCELoss(reduction='mean')
    l = bce(torch.squeeze(y1, dim=1), t1)
    print(l)
