"""
raw2rgb
ablation study
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .NAFNet_arch import SimpleGate, LayerNorm2d

class EndBlock(nn.Module):
  def __init__(self, ch_in, ch_out, ch_mid):
    super().__init__()
    self.ch_in = ch_in
    self.ch_out = ch_out

    self.conv1 = Conv(ch_in, ch_out, kernel_size=3, padding=1, mid_channels=ch_mid)
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


class NAFBlockV3(nn.Module):
  def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., inverted=False, sca=True, use_layernorm=True, **kwargs):
    super().__init__()
    dw_channel = c * DW_Expand if not inverted else c
    self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
    self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    
    # Simplified Channel Attention
    self.sca = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
    )

    # SimpleGate
    self.sg = SimpleGate()

    ffn_channel = FFN_Expand * c if not inverted else c
    self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    self.norm1 = LayerNorm2d(c) if use_layernorm else nn.Identity()
    self.norm2 = LayerNorm2d(c) if use_layernorm else nn.Identity()

    self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
    self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
  
  def forward(self, inp):
    x = inp

    x = self.norm1(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.sg(x)
    x = x * self.sca(x)
    x = self.conv3(x)

    x = self.dropout1(x)

    y = inp + x * self.beta

    x = self.conv4(self.norm2(y))
    x = self.sg(x)
    x = self.conv5(x)

    x = self.dropout2(x)

    return y + x * self.gamma

class Conv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=1, mid_channels=None, rate_mid=None):
    super().__init__()
    ch_max = max(in_channels, out_channels)
    if mid_channels == None and rate_mid == None:
      raise ValueError()
    elif mid_channels == None:
      mid_channels = int( ((kernel_size**2) * (in_channels*out_channels - ch_max)) / (in_channels+out_channels) * rate_mid)
    
    convs = [
      nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
      nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    ]

    if kernel_size > 1:
      conv_dw = nn.Conv2d(in_channels=ch_max, out_channels=ch_max, kernel_size=kernel_size, padding=padding, stride=1, groups=ch_max, bias=True)
      if in_channels > out_channels:
        convs = [conv_dw] + convs
      else:
        convs = convs + [conv_dw]
      self.convs = nn.Sequential(*convs)

  def forward(self, x):
    return self.convs(x)

class NAFBlockV3_Type1(nn.Module):
  """
  Middle block
  sca: o
  p.conv + p.conv + dw.conv
  """
  def __init__(self, c, c_mid=None, rate_mid=None, drop_out_rate=0., kernel_size=3, padding=1, use_layernorm=True, **kwargs):
    super().__init__()
    if c_mid == None and rate_mid == None:
      c_mid = c

    self.conv1 = Conv(c, c, kernel_size=3, padding=1, mid_channels=c_mid, rate_mid=rate_mid)
    self.conv2 = Conv(c//2, c, kernel_size=3, padding=1, mid_channels=c_mid, rate_mid=rate_mid)

    # Simplified Channel Attention
    self.sca = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=c//2, out_channels=c//2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
    )

    self.sg = SimpleGate()

    self.conv3 = Conv(c, c, kernel_size=3, padding=1, mid_channels=c_mid, rate_mid=rate_mid)
    self.conv4 = Conv(c//2, c, kernel_size=3, padding=1, mid_channels=c_mid, rate_mid=rate_mid)

    self.norm1 = LayerNorm2d(c) if use_layernorm else nn.Identity()
    self.norm2 = LayerNorm2d(c) if use_layernorm else nn.Identity()
    
    self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
    self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
  
  def forward(self, inp):
    x = inp

    x = self.norm1(x)
    x = self.conv1(x)
    x = self.sg(x)
    x = x * self.sca(x)
    x = self.conv2(x)

    x = self.dropout1(x)

    y = inp + x * self.beta

    x = self.norm2(y)
    x = self.conv3(x)
    x = self.sg(x)
    x = self.conv4(x)

    x = self.dropout2(x)

    return y + x * self.gamma

# class EndBlock()

class NAFBlockV3_nosca(NAFBlockV3):
  def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0, inverted=False, sca=True, use_layernorm=True, **kwargs):
    super().__init__(c, DW_Expand, FFN_Expand, drop_out_rate, inverted, sca, use_layernorm, **kwargs)
    
  def forward(self, inp):
    x = inp

    x = self.norm1(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.sg(x)
    # x = x * self.sca(x)
    x = self.conv3(x)

    x = self.dropout1(x)

    y = inp + x * self.beta

    x = self.conv4(self.norm2(y))
    x = self.sg(x)
    x = self.conv5(x)

    x = self.dropout2(x)

    return y + x * self.gamma

class NAFBlockV3_Type1_nosca(NAFBlockV3_Type1):
  def __init__(self, c, c_mid=None, rate_mid=None, drop_out_rate=0, kernel_size=3, padding=1, use_layernorm=True, **kwargs):
    super().__init__(c, c_mid, rate_mid, drop_out_rate, kernel_size, padding, use_layernorm, **kwargs)
  
  def forward(self, inp):
    x = inp

    x = self.norm1(x)
    x = self.conv1(x)
    x = self.sg(x)
    x = self.conv2(x)

    x = self.dropout1(x)

    y = inp + x * self.beta

    x = self.norm2(y)
    x = self.conv3(x)
    x = self.sg(x)
    x = self.conv4(x)

    x = self.dropout2(x)

    return y + x * self.gamma


class NAFNetV3_rgb(nn.Module):
  def __init__(self, input_shuffle=1, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], shuffle=1, inverted=False, middle_blk_type=0, sca=True, rate_mid=None, ch_mid=None, ending_blk_type=0, use_layernorm=True):
    super().__init__()
    self.input_shuffle = input_shuffle
    self.shuffle = shuffle
    self.in_ch = 3 # for rgb learning

    img_channel = self.in_ch * (shuffle**2)
    # self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
    # self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

    self.encoders = nn.ModuleList()
    self.decoders = nn.ModuleList()
    self.middle_blks = nn.ModuleList()
    self.ups = nn.ModuleList()
    self.downs = nn.ModuleList()

    print(f"inverted: {inverted}")

    if ending_blk_type == 1:
      self.intro = EndBlock(ch_in=img_channel, ch_out=width, ch_mid=ch_mid)
      self.ending = EndBlock(ch_in=width, ch_out=img_channel, ch_mid=ch_mid)
    else:
      self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
      self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)


    if middle_blk_type == 1:
      if sca == True:
        middle_blk_cls = lambda *args, **kwargs: NAFBlockV3_Type1(*args, use_layernorm=use_layernorm, **kwargs)
      else:
        middle_blk_cls = lambda *args, **kwargs: NAFBlockV3_Type1_nosca(*args, use_layernorm=use_layernorm, **kwargs)
    else:
      if sca == True:
        middle_blk_cls = lambda *args, **kwargs: NAFBlockV3(*args, use_layernorm=use_layernorm, **kwargs)
      else:
        middle_blk_cls = lambda *args, **kwargs: NAFBlockV3_nosca(*args, use_layernorm=use_layernorm, **kwargs)
    print(f"middle block type: {middle_blk_type}")

    chan = width
    for num in enc_blk_nums:
      self.encoders.append(
        nn.Sequential(*[middle_blk_cls(chan, inverted=inverted, c_mid=ch_mid, rate_mid=rate_mid) for _ in range(num)])
      )
      self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
      chan = chan * 2

    
    self.middle_blks = nn.Sequential(*[middle_blk_cls(chan, inverted=inverted, c_mid=ch_mid, rate_mid=rate_mid) for _ in range(middle_blk_num)])

    for num in dec_blk_nums:
      self.ups.append(nn.Sequential(
        nn.Conv2d(chan, chan*2, 1, bias=False),
        nn.PixelShuffle(2)
      ))
      chan = chan // 2
      self.decoders.append(nn.Sequential(*[middle_blk_cls(chan, inverted=inverted) for _ in range(num)]))
    
    self.padder_size = 2 ** len(self.encoders)
  
  def forward(self, inp):
    #inp = nn.PixelUnshuffle(self.input_shuffle)(inp) << This shuffle is for RAW so decrepated it
    inp = nn.PixelUnshuffle(self.shuffle)(inp)

    B, C, H, W = inp.shape
    inp = self.check_image_size(inp)

    x = self.intro(inp)

    encs = []

    for encoder, down in zip(self.encoders, self.downs):
      x = encoder(x)
      encs.append(x)
      x = down(x)
    
    x = self.middle_blks(x)
    
    for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
      x = up(x)
      x = x + enc_skip
      x = decoder(x)
    
    x = self.ending(x)
    x = x + inp

    x = x[:, :, :H, :W]
    x = nn.PixelShuffle(self.shuffle)(x)
    #x = nn.PixelShuffle(self.input_shuffle)(x) << This shuffle is for RAW so decrepated it

    return x

  def check_image_size(self, x):
    _, _, h, w = x.size()
    mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
    mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
    return x

# class EndBlock(nn.Module):
#   def __init__(self, ch_in, ch_out, ch_mid=None, rate_mid=None):
#     if rate_mid == None and ch_mid == None:
#       ch_mid = max([ch_in, ch_out])
#     self.ch_in = ch_in
#     self.ch_out = ch_out

#     self.conv1 = 

