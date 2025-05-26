import os
import os.path as osp
import mat73
import numpy as np
import scipy.signal as sg
import scipy.io as sio
import torch
from torch.utils import data as data

from pathlib import Path
from basicsr.utils import FileClient, scandir, imfrombytes, imfrombytes_np

class SiddBenchmarkDataset(data.Dataset):
  """
  for benchmark, raw2srgb
  with 
  """

  def __init__(self, opt):
    super(SiddBenchmarkDataset, self).__init__()
    self.opt = opt
    self.file_client = None
    self.io_backend_opt = opt['io_backend']
    self.io_backend_type = self.io_backend_opt.pop('type')
    
    self.gt_folder = opt['dataroot_gt']
    self.lq_folder = opt['dataroot_lq']
    self.gt_raw_folder = opt['dataroot_gt_raw']
    self.benchmark_folder = opt['benchmark_folder']
    self.benchmark_images = sorted(os.listdir(self.benchmark_folder), key=lambda x: int(x.split('_')[0]))


    self.filename_tmpl = opt.get('filename_tmpl', '{}')

    if self.io_backend_type != 'disk':
      raise NotImplementedError()
    else:
      self.paths = self._paired_paths()
  
  def __getitem__(self, index):
    if self.file_client is None:
      self.file_client = FileClient(self.io_backend_type, **self.io_backend_opt)
    
    scale = self.opt['scale']

    gt_path = self.paths[index]['gt_path']
    # load images
    img_bytes = self.file_client.get(gt_path)
    img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)

    lq_path = self.paths[index]['lq_path']
    img_bytes = self.file_client.get(lq_path)
    img_lq = imfrombytes_np(img_bytes, (256, 256), np.float32)

    gt_raw_path = self.paths[index]['gt_raw_path']
    img_bytes = self.file_client.get(gt_raw_path)
    img_gt_raw = imfrombytes_np(img_bytes, (256, 256), np.float32)

    img_gt = torch.tensor(img_gt, dtype=torch.float32).unsqueeze(0)
    img_lq = torch.tensor(img_lq, dtype=torch.float32).unsqueeze(0)
    img_gt_raw = torch.tensor(img_gt_raw, dtype=torch.float32).unsqueeze(0)


    if img_lq.ndim == 3:
      # print('lq')
      img_lq = img_lq.unsqueeze(0)

    if img_gt.ndim == 3:
      # print('gt')
      img_gt = img_gt.unsqueeze(0)

    if img_gt_raw.ndim == 3:
      # print('gt')
      img_gt_raw = img_gt_raw.unsqueeze(0)

    meta_path = self.paths[index]['meta_path']

    return {
      'lq': img_lq,
      'gt': img_gt, 
      'gt_raw': img_gt_raw,
      'lq_path': lq_path, 
      'gt_path': gt_path,
      'meta_path': meta_path
    }
  
  def __len__(self):
    return len(self.paths)

  def _paired_paths(self):
    input_folder = self.lq_folder
    gt_folder = self.gt_folder
    gt_raw_folder = self.gt_raw_folder
    benchmark_folder = self.benchmark_folder

    gt_paths = list(scandir(gt_folder))
    paths = []
    for idx in range(len(gt_paths)):
      gt_path = gt_paths[idx]
      gt_basename, ext = osp.splitext(osp.basename(gt_path))
      _, patch_idx = gt_basename.split('_')
      input_basename = f'ValidationBlocksRaw_{patch_idx}.raw'
      gt_raw_basename = input_basename

      input_path = osp.join(input_folder, input_basename)
      gt_path = osp.join(gt_folder, gt_path)
      gt_raw_path = osp.join(gt_raw_folder, input_basename)
      image_index = int(patch_idx) // 32
      image_name = self.benchmark_images[image_index]
      image_folder = Path(benchmark_folder).joinpath(image_name)
      metadata_fname = next((path for path in os.listdir(image_folder) if 'METADATA' in path), None)
      metadata_path = image_folder.joinpath(metadata_fname)
      paths.append({
        'gt_path': gt_path,
        'lq_path': input_path,
        'gt_raw_path': gt_raw_path,
        'meta_path': metadata_path
      })
    return paths
  
  @staticmethod
  def preproc(normalized_image, metadata_path):
    """
    isp pipeline before denoising in bayer domain
    """
    metadata = sio.loadmat(metadata_path)
    meta = [(key, value) for key, value in zip( metadata['metadata'][0].dtype.names, metadata['metadata'][0][0])]
    meta = dict(meta)

    cfa_pattern = SiddBenchmarkDataset.get_pattern_list(metadata_path)
    whitelevel = meta['AsShotNeutral'][0]

    wb_out = SiddBenchmarkDataset.white_balance(normalized_image, whitelevel, cfa_pattern)
    return wb_out

  @staticmethod
  def postproc(input, metadata_path):
    """
    isp pipeline after denoising in bayer domain, before demosaic
    """
    metadata = sio.loadmat(metadata_path)
    meta = [(key, value) for key, value in zip( metadata['metadata'][0].dtype.names, metadata['metadata'][0][0])]
    meta = dict(meta)
    pattern = SiddBenchmarkDataset.get_pattern(metadata_path)
    color_matrix = meta['ColorMatrix2'].astype(np.float64)

    temp = np.clip(input, 0, 1)
    temp = np.round(np.clip(input*(2**16), 0, (2**16)-1)).astype(np.uint16)
    demosaic_int = SiddBenchmarkDataset.demosaic_int16(temp, pattern)
    demosaic_out = demosaic_int.astype(np.float32)/(2**16)

    xyz_image = SiddBenchmarkDataset.apply_color_space_transform(demosaic_out, color_matrix_2=color_matrix)
    srgb_out = SiddBenchmarkDataset.transform_xyz_to_srgb(xyz_image)
    return srgb_out

  @staticmethod
  def get_pattern(metadata_path):
    patterns = {
      'GP': 'bggr',
      'IP': 'rggb',
      'S6': 'grbg',
      'N6': 'bggr',
      'G4': 'bggr'
    }
    if 'S6' in metadata_path:
      return patterns['S6']
    elif 'GP' in metadata_path:
      return patterns['GP']
    elif 'N6' in metadata_path:
      return patterns['N6']
    elif 'IP' in metadata_path:
      return patterns['IP']
    elif 'G4' in metadata_path:
      return patterns['G4']
    else:
      return None
  
  @staticmethod
  def get_pattern_list(metadata_path):
    if 'S6' in metadata_path:
      return [1, 0, 2, 1]
    elif 'GP' in metadata_path:
      return [2, 1, 1, 0]
    elif 'N6' in metadata_path:
      return [2, 1, 1, 0]
    elif 'IP' in metadata_path:
      return [0, 1, 1, 2]
    elif 'G4' in metadata_path:
      return [2, 1, 1, 0]
    else:
      return None
  
  @staticmethod
  def white_balance(normalized_image, as_shot_neutral, cfa_pattern):
    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    white_balanced_image = np.zeros(normalized_image.shape)
    for i, idx in enumerate(idx2by2):
      idx_y = idx[0]
      idx_x = idx[1]
      white_balanced_image[idx_y::step2, idx_x::step2] = normalized_image[idx_y::step2, idx_x::step2] / as_shot_neutral[cfa_pattern[i]]
    white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image
  
  @staticmethod
  def demosaic_int16(image, pattern):
    mask_r, mask_gr, mask_gb, mask_b = get_bayer_mask(image.shape, pattern)
    h_g = np.array([
      [ 0,  0, -1,  0,  0],
      [ 0,  0,  2,  0,  0],
      [-1,  2,  4,  2, -1],
      [ 0,  0,  2,  0,  0],
      [ 0,  0, -1,  0,  0]])
    h_rb_h = np.array([
      [  0,  0, 0.5,  0,   0],
      [  0, -1,   0, -1,   0],
      [ -1,  4,   5,  4,  -1],
      [  0, -1,   0, -1,   0],
      [  0,  0, 0.5,  0,   0]])
    h_rb_v = h_rb_h.T
    h_rb_omni = np.array([
      [0, 0, -1.5, 0, 0],
      [0, 2, 0, 2, 0],
      [-1.5, 0, 6, 0, -1.5],
      [0, 2, 0, 2, 0],
      [0, 0, -1.5, 0, 0]])
    img_pad = np.pad(image, 2, 'reflect').astype(np.int32)
  
    # green interpolation
    g_interp = np.right_shift(sg.convolve2d(img_pad, h_g, 'valid'), 3)
    g = image * (mask_gr+mask_gb) + np.clip(g_interp, 0, (2**16)) * (mask_r+mask_b)
    
    # red, blue interpolation
    rb_interp_h = np.right_shift(sg.convolve2d(img_pad, (h_rb_h*2).astype(np.int32), 'valid'), 4)
    rb_interp_v = np.right_shift(sg.convolve2d(img_pad, (h_rb_v*2).astype(np.int32), 'valid'), 4)
    rb_interp_hv = np.right_shift(sg.convolve2d(img_pad, (h_rb_omni*2).astype(np.int32), 'valid'), 4)

    r = mask_gr * rb_interp_h + mask_gb * rb_interp_v + mask_b * rb_interp_hv + mask_r * image
    b = mask_gb * rb_interp_h + mask_gr * rb_interp_v + mask_r * rb_interp_hv + mask_b * image

    return np.dstack([r, g, b])

  @staticmethod
  def apply_color_space_transform(rgb_image, color_matrix_1=None, color_matrix_2=None):
    color_matrix = color_matrix_2.reshape(3, 3)
    color_matrix = color_matrix / np.sum(color_matrix, axis=1, keepdims=True)
    color_matrix = np.linalg.inv(color_matrix)

    xyz_image = rgb_image @ (color_matrix.T)
    xyz_image = np.clip(xyz_image, 0, 1)
    
    return xyz_image

  @staticmethod
  def transform_xyz_to_srgb(xyz_image):
    M = np.array([
      [3.2404542, -1.5371385, -0.4985314],
      [-0.9692660, 1.8760108, 0.0415560],
      [0.0556434, -0.2040259, 1.0572252]
    ])

    rgb_linear = xyz_image @ (M.T)
    rgb_linear = np.clip(rgb_linear, 0, 1)

    mask_lin = rgb_linear < 0.0031308
    rgb_gamma = 1.055 * np.power(rgb_linear, 1/2.4) - 0.055
    rgb_lin = 12.92 * rgb_linear
    srgb_image = rgb_gamma * (1-mask_lin) + rgb_lin * mask_lin
    srgb_image = np.clip(srgb_image, 0, 1)
    return srgb_image

def get_bayer_mask(shape, pattern='rggb'):
  r_idx = pattern.index('r')
  b_idx = pattern.index('b')
  gr_idx = r_idx-1 if r_idx%2 else r_idx+1
  gb_idx = b_idx-1 if b_idx%2 else b_idx+1

  _pattern = np.array(['__' for _ in range(4)])
  _pattern[[r_idx, gr_idx, gb_idx, b_idx]] = ['r', 'gr', 'gb', 'b']
  pattern = list(_pattern)

  channels = dict((channel, np.zeros(shape)) for channel in ['r', 'gr', 'gb', 'b'])
  for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
    channels[channel][y::2, x::2] = 1
    
  return tuple(channels[c].astype(np.int32) for c in ['r', 'gr', 'gb', 'b'])
