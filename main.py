from network import *
import torch
import torch.nn as nn
data = torch.rand(1,3,224,224)
model = resnet50(3)
output = model(data)
print(output) 