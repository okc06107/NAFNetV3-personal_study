import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data as data
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.data_util import paired_paths_from_lmdb, paired_paths_from_folder

from pathlib import Path

from basicsr.utils import FileClient, imfrombytes_np, imfrombytes_bitsplit, padding

class PairedImageRawDataset(data.Dataset):
  def __init__(self, opt):
    super(PairedImageRawDataset, self).__init__()
    self.opt = opt
    self.file_client = None
    self.io_backend_opt = opt['io_backend']
    
    self.gt_folder = opt['dataroot_gt']
    self.lq_folder = opt['dataroot_lq']
    if 'filename_tmpl' in opt:
        self.filename_tmpl = opt['filename_tmpl']
    else:
        self.filename_tmpl = '{}'
    self.io_backend_type = self.io_backend_opt.pop('type')
    
    if self.io_backend_type == 'lmdb':
      self.io_backend_opt['db_paths'] = [
        Path(self.lq_folder).joinpath('lsb.lmdb'),
        Path(self.lq_folder).joinpath('msb.lmdb'),
        Path(self.gt_folder).joinpath('lsb.lmdb'),
        Path(self.gt_folder).joinpath('msb.lmdb')
        # osp.join(self.lq_folder, 'lsb.lmdb'),
        # osp.join(self.lq_folder, 'msb.lmdb'),
        # osp.join(self.gt_folder, 'lsb.lmdb'),
        # osp.join(self.gt_folder, 'msb.lmdb')
      ]
      self.io_backend_opt['client_keys'] = [
        'lq_lsb',
        'lq_msb',
        'gt_lsb',
        'gt_msb'
      ]
      self.paths = paired_paths_from_lmdb(
        [osp.join(self.lq_folder, 'lsb.lmdb'),osp.join(self.gt_folder, 'lsb.lmdb')],
        ['lq', 'gt']
      )
    elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
      raise NotImplementedError()
    elif self.io_backend_type == 'disk':
      # self.lq_files = os.listdir(self.lq_folder)
      # self.gt_files = os.listdir(self.gt_folder)
      # assert (len(self.lq_files) == len(self.gt_files))
      self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
    else:
      raise NotImplementedError()

    self.mean = None
    self.std = None
  
  def __getitem__(self, index):
    if self.file_client is None:
      self.file_client = FileClient(self.io_backend_type, **self.io_backend_opt)
    
    scale = self.opt['scale']
    
    gt_path = self.paths[index]['gt_path']
    if self.io_backend_type == 'lmdb':
      gt_lsb_img_bytes = self.file_client.get(gt_path, 'gt_lsb')
      gt_msb_img_bytes = self.file_client.get(gt_path, 'gt_msb')
      img_gt = imfrombytes_bitsplit(gt_lsb_img_bytes, gt_msb_img_bytes)
    else:
      img_bytes = self.file_client.get(gt_path)
      if self.opt['phase'] == 'train':
        img_gt = imfrombytes_np(img_bytes, (512, 512), np.float32)
      else:
        img_gt = imfrombytes_np(img_bytes, (256, 256), np.float32)

    lq_path = self.paths[index]['lq_path']
    if self.io_backend_type == 'lmdb':
      lq_lsb_img_bytes = self.file_client.get(lq_path, 'lq_lsb')
      lq_msb_img_bytes = self.file_client.get(lq_path, 'lq_msb')
      img_lq = imfrombytes_bitsplit(lq_lsb_img_bytes, lq_msb_img_bytes)
    else:
      img_bytes = self.file_client.get(lq_path)
      if self.opt['phase'] == 'train':
        img_lq = imfrombytes_np(img_bytes, (512, 512), np.float32)
      else:
        img_lq = imfrombytes_np(img_bytes, (256, 256), np.float32)

    if self.opt['phase'] == 'train':
      # augmentation
      gt_size = self.opt['gt_size']
      scale = self.opt['scale']

      img_gt, img_lq = padding(img_gt, img_lq, gt_size)
      img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path="")
      img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

    # make tensor
    img_gt = torch.tensor(img_gt, dtype=torch.float32).unsqueeze(0)
    img_lq = torch.tensor(img_lq, dtype=torch.float32).unsqueeze(0)

    if self.mean is not None or self.std is not None:
      raise NotImplementedError()

    return {
      'lq': img_lq,
      'gt': img_gt,
      'lq_path': lq_path,
      'gt_path': gt_path
    }

  def __len__(self):
    return len(self.paths)