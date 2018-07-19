import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from .resnet import resnet50
from .deform_conv import DeformConv2D

class Model(nn.Module):
  def __init__(self, last_conv_stride=2):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)
    # Divserse transform
    self.conv1 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)

    self.bn1 = nn.BatchNorm2d(2048)
    self.bn2 = nn.BatchNorm2d(2048)
    self.bn3 = nn.BatchNorm2d(2048)
    self.bn4 = nn.BatchNorm2d(2048)

    self.relu = nn.ReLU(inplace=True)

    # Deformable 2D
    # self.offsets = nn.Conv2d(2048, 18, kernel_size=3, padding=1)
    # self.conv4 = DeformConv2D(2048, 2048, kernel_size=3, padding=1)
    # self.bn4 = nn.BatchNorm2d(2048)

  # def forward(self, x, m):
  #   # x shape [N, C, H, W], m shape [N, 256, 128]
  #   x = self.base(x)
  #   N, C, H, W = x.shape
  #   # Deformable Conv
  #   # import pdb
  #   # pdb.set_trace()
  #   # offsets = self.offsets(x)
  #   # x = F.relu(self.conv4(x, offsets))
  #   # x = self.bn4(x)

  #   # m -> [N, 1, H, W]
  #   x = F.avg_pool2d(x, x.size()[2:])
  #   # shape [N, C]
  #   x = x.view(x.size(0), -1)


  # def forward(self, x, m):
  #   # x shape [N, C, H, W], m shape [N, 256, 128]
  #   N, C, H, W = x.shape
  #   # m -> [N, 1, H, W]
  #   # m = Variable(m)
  #   m = m.view((N, 1, H, W))
  #   m = m.expand((N, C, H, W))
  #   # m = m.expand((N, C, H, W))
  #   x = x * m 
  #   x = self.base(x)
  #   x = F.avg_pool2d(x, x.size()[2:])
  #   # shape [N, C]
  #   x = x.view(x.size(0), -1)

  # def forward(self, x, m):
  #   # x shape [N, C, H, W], m shape [N, 256, 128]
  #   x = self.base(x)
  #   N, C, H, W = x.shape
  #   # m -> [N, 1, H, W]
  #   # m = Variable(m)
  #   m = F.avg_pool2d(m, (16,16))
  #   # area normalization
  #   s = torch.sum(m,1)
  #   s = torch.sum(s,1)
  #   s = s.view((N, 1))
  #   s = s.expand((N, C))
  #   # end
  #   m = m.view((N, 1, H, W))
  #   m = m.expand((N, C, H, W))
  #   # m = m.expand((N, C, H, W))
  #   x = x * m 
  #   x = F.avg_pool2d(x, x.size()[2:])
  #   # shape [N, C]
  #   x = x.view(x.size(0), -1)
  #   # area nml
  #   x = torch.div(x , s)

    # return x

  def forward(self, x, m):
    # x shape [N, C, H, W], m shape [N, 256, 128]
    x = self.base(x)
    x1 = self.relu(self.bn1(self.conv1(x)))
    x2 = self.relu(self.bn1(self.conv2(x)))
    x3 = self.relu(self.bn1(self.conv3(x)))
    x4 = self.relu(self.bn1(self.conv4(x)))
    
    x1 = F.avg_pool2d(x1, x1.size()[2:])
    x1 = x1.view(x1.size(0), -1)
    x2 = F.avg_pool2d(x2, x2.size()[2:])
    x2 = x2.view(x2.size(0), -1)
    x3 = F.avg_pool2d(x3, x3.size()[2:])
    x3 = x3.view(x3.size(0), -1)
    x4 = F.avg_pool2d(x4, x4.size()[2:])
    x4 = x4.view(x4.size(0), -1)

    x = F.avg_pool2d(x, x.size()[2:])
    x = x.view(x.size(0), -1)
    return torch.stack((x, x1, x2, x3, x4),2)
