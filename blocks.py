import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class  bottleneck_block(nn.Module):
    def __init__(self,in_num,out_num,mid_num = 64,stride=1,downsampling = None):
        super(bottleneck_block,self).__init__()
        self.conv1 = nn.Conv2d(in_num,mid_num,stride = 1 ,kernel_size = 1,padding = 0)
        self.bn_proj = nn.BatchNorm2d(out_num)
        self.bn = nn. BatchNorm2d(mid_num)
        
        if downsampling == None:
            self.conv2 = nn.Conv2d(mid_num,mid_num,stride = 1,kernel_size = 3,padding = 1)
        else:
            self.conv2 = nn.Conv2d(mid_num,mid_num,stride = 2,kernel_size = 3,padding = 1)
        
        self.conv3 = nn.Conv2d(mid_num,out_num,stride = 1 , kernel_size = 1,padding = 0)
        if in_num == out_num:
            self.proj = lambda x:x
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_num,out_num,stride = stride,kernel_size =1 ,padding= 0)
            )
    def forward(self,x):
        x_shortcut = self.proj(x)
        x_shortcut = self.bn_proj(x_shortcut)
        print("x_shortcut Shape: ", x_shortcut.shape)
          
        x_out = self.conv1(x)
        x_out = self.bn(x_out)
        x_out = F.relu(x_out)
        x_out = self.conv2(x_out)
        x_out = self.bn(x_out)
        x_out = F.relu(x_out)
        x_out = self.conv3(x_out)
        print("x_out Shape: ", x_out.shape)
        x_out = x_out+x_shortcut
        x_out = F.relu(x_out)
      
        return x_out
