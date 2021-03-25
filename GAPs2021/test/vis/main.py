import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

res = np.load('history.npy')
df_cm = pd.DataFrame(res[res.shape[0] -1 ], index = [i for i in "ABCDEFGH"],
                  columns = [i for i in "ABCDEFGH"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()


# import torch.nn as nn
# import torch
#
# # x = torch.rand((4,8))
# # y = torch.randint(8, (4,))
# x = torch.tensor([[[[0.3793, 0.3034, 0.5449],
#           [0.9949, 0.5025, 0.9944],
#           [0.2574, 0.0197, 0.5068]],
#
#          [[0.3567, 0.9957, 0.3658],
#           [0.8256, 0.6944, 0.3829],
#           [0.4983, 0.7530, 0.7513]],
#
#          [[0.0290, 0.2186, 0.9086],
#           [0.2926, 0.7689, 0.5136],
#           [0.5261, 0.3140, 0.7821]],
#
#          [[0.2951, 0.5755, 0.4455],
#           [0.5463, 0.5445, 0.3220],
#           [0.6803, 0.1969, 0.5451]]]])
# y = torch.tensor([[[3, 2, 0],
#          [2, 1, 0],
#          [1, 3, 2]]])
# print(x.shape)
# print(y.shape)
#
# logsoftmax = nn.LogSoftmax(dim=1)
# nll = nn.NLLLoss()
# crossentropy = nn.CrossEntropyLoss(reduction='none')
# print(nll(logsoftmax(x), y))
# print(crossentropy(x, y))