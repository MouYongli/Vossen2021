import torchvision

def VGG16(pretrained=True):
    return torchvision.models.vgg16(pretrained=pretrained)



