import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'conv_3x3' : lambda C, stride, affine: Conv(C, C, 3, stride, 1, affine=affine),
  'conv_5x5' : lambda C, stride, affine: Conv(C, C, 5, stride, 2, affine=affine),
}
class Conv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(Conv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm1d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    n, c, h= x.size()
    if self.stride>1:
      h = (h+1) // self.stride
    if x.is_cuda:
      with torch.cuda.device(x.get_device()):
        padding = torch.cuda.FloatTensor(n, c, h).fill_(0)
    else:
      padding = torch.FloatTensor(n, c, h).fill_(0)
    return padding    

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm1d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
    out = self.bn(out)
    return out

