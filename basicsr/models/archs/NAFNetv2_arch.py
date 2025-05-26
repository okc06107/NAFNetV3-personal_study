import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .NAFNet_arch import SimpleGate, LayerNorm2d

class NAFNetv2(nn.Module):
  def __init__(self, ch=90, sc0=2, sc1=0, rate_mid=0.16, middle_blk_num=9):
    super().__init__()
    self.sc0 = sc0 # pixel unshuffle factor
    self.sc0Pow = int(2**self.sc0)
    self.sc1 = sc1 # intro/ending block number

    nch_in = int(4 * self.sc0Pow * self.sc0Pow)
    nch_out = int(4 * self.sc0Pow * self.sc0Pow)
    print('nch_in: ', nch_in)

    enc_blk_nums = [1] * self.sc1
    dec_blk_nums = [1] * self.sc1
    
    width = int(ch)

    middle_blk_num = int(middle_blk_num - self.sc1*2)

    self.intro = EndBlock(nch_in, width, rate_mid)
    self.ending = EndBlock(width, nch_out, rate_mid)

    self.encoders = nn.ModuleList()
    self.decoders = nn.ModuleList()
    self.middle_blks = nn.ModuleList()
    self.ups = nn.ModuleList()
    self.downs = nn.ModuleList()

    chan = width
    if self.sc1 > 0:
      for num in enc_blk_nums:
        self.encoders.append(
          nn.Sequential(*[MidBlock(chan, rate_mid) for _ in range(num)])
        )
        self.downs.append(
          nn.PixelUnshuffle(2)
        )
        chan = chan * 4
    
    self.middle_blks = nn.Sequential(*[MidBlock(chan, rate_mid) for _ in range(middle_blk_num)])

    if self.sc1 > 0:
      for num in dec_blk_nums:
        self.ups.append(
          nn.PixelShuffle(2)
        )
        self.decoders.append(
          nn.Sequential(*[MidBlock(chan, rate_mid) for _ in range(num)])
        )
    
    self.padder_size = int(2**(self.sc0 + self.sc1))
    self._init_weights()

  def forward(self, inp):
    inp = nn.PixelUnshuffle(2)(inp) # bayer pattern split (r, gr, gb, b)
    B, C, H, W = inp.shape
    inp = self.check_image_size(inp)

    if self.sc0 > 0:
      inp = nn.PixelUnshuffle(self.sc0Pow)(inp)
    x = self.intro(inp)

    if self.sc1 > 0:
      encs = []
      for encoder, down in zip(self.encoders, self.downs):
        x = encoder(x)
        encs.append(x)
        x = down(x)
    
    x = self.middle_blks(x)

    if self.sc1 > 0:
      for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
        x = up(x)
        # x = x + enc_skip # remove skip connection
        x = decoder(x)
    
    x = self.ending(x)
    if self.sc0 > 0:
      x = nn.PixelShuffle(self.sc0Pow)(x)
    
    x = x[:, :, :H, :W]
    x = nn.PixelShuffle(2)(x)

    return x

  def check_image_size(self, x):
    _, _, h, w = x.size()
    mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
    mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
    return x

  def _init_weights(self, init_type='he_in', init_gain=1):
    def init_func(m):
      classname = m.__class__.__name__
      if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'he_in':
          temp = m.weight.data
          szt = temp.shape
          std_ = init_gain * math.sqrt(1. / (szt[1] * szt[2] * szt[3]))
          init.normal_(m.weight.data, std=std_)
        else:
          raise NotImplementedError(f'initialization meethod [{init_type}] is not implemented')
        if hasattr(m, 'bias') and m.bias is not None:
          init.constant_(m.bias.data, 1e-8)
    def init_func_skipInit(m):
      if hasattr(m, 'init_a'):
        init.constant_(m.conv1.weight.data, 1e-8)
    
    self.apply(init_func)
    self.apply(init_func_skipInit)

class EndBlock(nn.Module):
  def __init__(self, ch_in, ch_out, rate_mid=1):
    super().__init__()
    self.ch_in = ch_in
    self.ch_out = ch_out

    self.conv1 = DwSepConv(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, rate_mid=rate_mid)
    self.beta = nn.Parameter(torch.zeros((1, ch_out, 1, 1)), requires_grad=True)
  
  def forward(self, inp):
    x = self.conv1(inp)
    x = x * self.beta

    if self.ch_out > self.ch_in:
      n_rep = math.ceil(x.shape[1]/inp.shape[1])
      res = torch.tile(inp, (1, n_rep, 1, 1))
    else:
      res = inp
    res = res[:, :self.ch_out, :, :]

    y = x + res
    return y

class MidBlock(nn.Module):
  def __init__(self, c, rate_mid=1):
    super().__init__()
    self.norm1 = LayerNorm2d(c)
    self.conv1 = DwSepConv(in_channels=c, out_channels=c, kernel_size=3, padding=1, rate_mid=rate_mid)
    self.sg1 = SimpleGate()
    self.conv2 = DwSepConv(in_channels=c//2, out_channels=c, kernel_size=3, padding=1, rate_mid=rate_mid)
    self.beta = SkipInitMult(c)
    
    self.norm2 = LayerNorm2d(c)
    self.conv3 = DwSepConv(in_channels=c, out_channels=c, kernel_size=3, padding=1, rate_mid=rate_mid)
    self.sg2 = SimpleGate()
    self.conv4 = DwSepConv(in_channels=c//2, out_channels=c, kernel_size=3, padding=1, rate_mid=rate_mid)
    self.gamma = SkipInitMult(c)
  
  def forward(self, inp):
    x = inp
    x = self.norm1(x)
    x = self.conv1(x)
    x = self.sg1(x)
    x = self.conv2(x)
    x = self.beta(x)
    y = inp + x

    x = self.norm2(y)
    x = self.conv3(x)
    x = self.sg2(x)
    x = self.conv4(x)
    x = self.gamma(x)
    y = y + x
    return y

class DwSepConv(nn.Module):
  """
  Depthwise separable convolution
  """
  def __init__(self, in_channels, out_channels, kernel_size, padding=1, rate_mid=1):
    super().__init__()

    ch_max = max(in_channels, out_channels)
    ch_mid = int( ((kernel_size**2) * (in_channels*out_channels - ch_max)) / (in_channels+out_channels) * rate_mid) # why??

    convs = [
      nn.Conv2d(in_channels=in_channels, out_channels=ch_mid, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
      nn.Conv2d(in_channels=ch_mid, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    ]

    if kernel_size > 1:
      conv_dw = [nn.Conv2d(in_channels=ch_max, out_channels=ch_max, kernel_size=kernel_size, padding=padding, stride=1, groups=ch_max, bias=True)]
      if in_channels > out_channels:
        convs = conv_dw + convs
      else:
        convs = convs + conv_dw
    
    self.convs = nn.Sequential(*convs)
  
  def forward(self, x):
    y = self.convs(x)
    return y

class SkipInitMult(nn.Module):
  def __init__(self, c, bias=False):
    super().__init__()
    self.init_a = False
    self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=c, bias=bias)
  
  def forward(self, x):
    y = self.conv1(x)
    return y


if __name__ == "__main__":
  pass