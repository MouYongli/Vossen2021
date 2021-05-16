from modeling.backbone import resnet, xception, drn, mobilenet, efficientnet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'efficientnet':
        return efficientnet.get_encoder("efficientnet-b7", in_channels=3, depth=5, weights=None)
    else:
        raise NotImplementedError
