from models import FCN32s
from datasets import GapsDataset
import torch
from torch.utils.data import DataLoader

model = FCN32s(n_class=8)
# for key in model.state_dict().keys():
#     print(key)
# import torch
# x = torch.rand((4, 1, 512, 512))
# y = model(x)

train_loader = DataLoader(GapsDataset(split='train'), batch_size=8, shuffle=True)
val_loader = DataLoader(GapsDataset(split='valid-test'), batch_size=8, shuffle=False)

train_iter = iter(train_loader)
images, labels = next(train_iter)
print(images.shape)
print(labels.dtype)

import torch.nn as nn
lossfunc = nn.CrossEntropyLoss()

