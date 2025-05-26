import numpy as np
import cv2
import os
import mat73
import scipy.io as sio
from multiprocessing import Pool
from os import path as osp
from pathlib import Path
from tqdm import tqdm

from basicsr.utils import scandir_SIDD, scandir
# from basicsr.utils.lmdb_util import make_lmdb_from_np
from basicsr.utils.lmdb_util import make_lmdb_from_imgs_bitsplit


def make_trainset():
  opt = {}
  opt['n_thread'] = 12
  # opt['compression_level'] = 3


  opt['input_folder'] = 'D:/datasets/SIDD/SIDD_Medium/SIDD_Medium_Raw_Parts/SIDD_Medium_Raw/Data'
  opt['save_folder'] = 'D:/datasets/SIDD/SIDD_Medium/SIDD_Medium_Raw_Parts/lq_raw_crops'

  opt['crop_size'] = 512
  opt['step'] = 384
  opt['thresh_size'] = 0
  opt['keywords'] = '_NOISY_RAW'
  # extract_subimages(opt)
  # create_lmdb(opt)

  opt['save_folder'] = 'D:/datasets/SIDD/SIDD_Medium/SIDD_Medium_Raw_Parts/gt_raw_crops'
  opt['keywords'] = '_GT_RAW'
  # extract_subimages(opt)
  create_lmdb(opt)


def extract_subimages(opt):
  input_folder = opt['input_folder']
  save_folder = opt['save_folder']
  if not osp.exists(save_folder):
    os.makedirs(save_folder)
    print(f'mkdir {save_folder} ...')
  else:
    print(f'Folder {save_folder} already exists. Exit.')
    # sys.exit(1)

  img_list = list(scandir_SIDD(Path(input_folder), keywords=opt['keywords'], recursive=True, full_path=True))[:12]

  pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
  pool = Pool(opt['n_thread'])
  for path in img_list:
    pool.apply_async(
      worker, args=(path, opt), callback=lambda arg: pbar.update(1))
  pool.close()
  pool.join()
  pbar.close()
  print('All processes done.')

def worker(path, opt):
  crop_size = opt['crop_size']
  step = opt['step']
  thresh_size = opt['thresh_size']

  img_name, extension = osp.splitext(osp.basename(path))
  img_name = img_name.replace(opt['keywords'], '')
  img = mat73.loadmat(path)['x']

  h, w = img.shape
  h_space = np.arange(0, h - crop_size + 1, step)
  if h - (h_space[-1] + crop_size) > thresh_size:
    h_space = np.append(h_space, h - crop_size)
  w_space = np.arange(0, w - crop_size + 1, step)
  if w - (w_space[-1] + crop_size) > thresh_size:
    w_space = np.append(w_space, w - crop_size)
  
  index = 0
  for x in h_space:
    for y in w_space:
      index += 1
      cropped_img = img[x:x + crop_size, y:y + crop_size, ...].astype(np.float32)
      # cropped_img = np.ascontiguousarray(cropped_img)
      # cv2.imwrite(
      #   os.path.join(opt['save_folder'], f'{img_name}_s{index:03d}.png'), 
      #   cropped_img)
      cropped_img.tofile(osp.join(opt['save_folder'], f'{img_name}_s{index:03d}.raw'))
  process_info = f'Processing {img_name} ...'
  return process_info

def create_lmdb(opt):
  save_folder = Path(opt['save_folder'])
  # lmdb_name = str(save_folder.name)+'.lmdb'
  lmdb_name = str(save_folder.name)+'_lmdb'
  lmdb_folder = save_folder.parents[0].joinpath(lmdb_name)
  suffix = 'raw'
  img_path_list = sorted(list(scandir(save_folder, suffix='raw', recursive=False)))
  keys = [img_path.split(f'.{suffix}')[0] for img_path in sorted(img_path_list)]
  # make_lmdb_from_imgs(save_folder, lmdb_folder, img_path_list, keys)
  # make_lmdb_from_np(str(save_folder), str(lmdb_folder), img_path_list, keys, shape=(512, 512), dtype=np.float32)
  make_lmdb_from_imgs_bitsplit(str(save_folder), str(lmdb_folder), img_path_list, keys, shape=(512, 512), dtype=np.float32)

def make_validset(wb=True):
  from isp.sidd_pipeline import white_balance, get_pattern, get_pattern_list
  lq_path = "/home/swhong/01_NAFNet/AIISP2024/ValidationNoisyBlocksRaw.mat"
  save_folder = "/data1/SIDD_raw/valid/valid_lq_raw_crops"
  lq = sio.loadmat(lq_path)
  lq_images = lq['ValidationNoisyBlocksRaw']
  _shape = lq_images.shape
  lq_images = lq_images.reshape(-1, *_shape[2:])

  benchmark_folder = "/home/swhong/SIDD_Benchmark_Data"
  benchmark_images = sorted(os.listdir(benchmark_folder), key=lambda x: int(x.split('_')[0]))

  print('making validation - input')
  for i in tqdm(range(len(lq_images))):
    img = lq_images[i]
    if wb == True:
      img_index = i // 32
      img_name = benchmark_images[img_index]
      img_folder = Path(benchmark_folder).joinpath(img_name)
      metadata_fname = next((path for path in os.listdir(img_folder) if 'METADATA' in path), None)
      metadata_path = img_folder.joinpath(metadata_fname)
      metadata = sio.loadmat(metadata_path)
      meta = [(key, value) for key, value in zip( metadata['metadata'][0].dtype.names, metadata['metadata'][0][0])]
      meta = dict(meta)

      cfa_pattern = get_pattern_list(str(metadata_path))
      whitelevel = meta['AsShotNeutral'][0]
      final_img = white_balance(img, whitelevel, cfa_pattern).astype(np.float32)
      if i < 5:
        print(final_img.shape)
    else:
      final_img = img

    final_img.tofile(osp.join(save_folder, f'ValidationBlocksRaw_{i}.raw'))

  gt_path = "/home/swhong/01_NAFNet/AIISP2024/ValidationGtBlocksRaw.mat"
  save_folder = "/data1/SIDD_raw/valid/valid_gt_raw_crops"
  gt = sio.loadmat(gt_path)
  gt_images = gt['ValidationGtBlocksRaw']
  _shape = gt_images.shape
  gt_images = gt_images.reshape(-1, *_shape[2:])

  print('making validation - gt')
  for i in tqdm(range(len(gt_images))):
    img = gt_images[i]
    if wb == True:
      img_index = i // 32
      img_name = benchmark_images[img_index]
      img_folder = Path(benchmark_folder).joinpath(img_name)
      metadata_fname = next((path for path in os.listdir(img_folder) if 'METADATA' in path), None)
      metadata_path = img_folder.joinpath(metadata_fname)
      metadata = sio.loadmat(metadata_path)
      meta = [(key, value) for key, value in zip( metadata['metadata'][0].dtype.names, metadata['metadata'][0][0])]
      meta = dict(meta)

      cfa_pattern = get_pattern_list(str(metadata_path))
      whitelevel = meta['AsShotNeutral'][0]
      final_img = white_balance(img, whitelevel, cfa_pattern).astype(np.float32)
      if i < 5:
        print(final_img.shape)
    else:
      final_img = img

    final_img.tofile(osp.join(save_folder, f'ValidationBlocksRaw_{i}.raw'))

def make_raw2rgb_validset(rgb_only=True):
  if rgb_only:
    gt_path = "/home/swhong/01_NAFNet/AIISP2024/ValidationGtBlocksSrgb.mat"
    save_folder = "/data1/SIDD_raw/valid/valid_gt_srgb_crops"
    gt = sio.loadmat(gt_path)
    gt_images = gt['ValidationGtBlocksSrgb']
    _shape = gt_images.shape
    gt_images = gt_images.reshape(-1, *_shape[2:])
    for i in tqdm(range(len(gt_images))):
      img = gt_images[i]
      img = np.ascontiguousarray(img)
      cv2.imwrite(osp.join(save_folder, f'ValidationBlocksSrgb_{i}.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 2])
  else:
    raise NotImplementedError()


if __name__ == "__main__":
  # make_trainset()
  make_validset(wb=False) # for SIDD, test and valid set are same.
  # make_raw2rgb_validset()