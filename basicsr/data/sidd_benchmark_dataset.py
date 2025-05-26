import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data as data
from basicsr.utils import FileClient, scandir, imfrombytes, img2tensor
from basicsr.data.data_util import paired_paths_from_lmdb

class SiddBenchmarkDataset(data.Dataset):
    """
    SIDD Benchmark Dataset for RGB images
    """

    def __init__(self, opt):
        super(SiddBenchmarkDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.io_backend_type = self.io_backend_opt.pop('type')
        
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_type == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        else:
            raise NotImplementedError(f'IO backend {self.io_backend_type} is not supported')
  
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_type, **self.io_backend_opt)
        
        scale = self.opt['scale']

        # Load gt and lq images
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                   bgr2rgb=True,
                                   float32=True)

        # Add batch dimension if needed
        if img_lq.dim() == 3:
            img_lq = img_lq.unsqueeze(0)
        if img_gt.dim() == 3:
            img_gt = img_gt.unsqueeze(0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
  
    def __len__(self):
        return len(self.paths)
