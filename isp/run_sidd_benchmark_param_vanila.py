import os
import csv
import numpy as np
import scipy.io as sio
import cv2
from os import path as osp
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
from basicsr.data import create_dataset
from basicsr.utils import make_exp_dirs

from basicsr.data.sidd_benchmark_dataset import SiddBenchmarkDataset
from basicsr.models.image_restoration_model import ImageRestorationModel
from isp.sidd_pipeline import white_balance, demosaic_matlab, apply_color_space_transform, transform_xyz_to_srgb, get_pattern, get_pattern_list

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from basicsr.metrics import calculate_psnr, calculate_ssim

from basicsr.train_param import make_yaml_parser, rendering_template
import torchinfo



def main(wb_first=False):
  parser = ArgumentParser()
  # parser.add_argument('-opt', type=str, default='/home/swhong/01_NAFNet/AIISP2024/options/test/SIDD/NAFNetv2-wb.yml')
  parser.add_argument('-opt')
  parser.add_argument('--template')
  args, remained_args = parser.parse_known_args()

  template_path = args.template
  with open(template_path, 'r') as file:
    template_content = file.read()
  parser = make_yaml_parser(template_content)
  args = parser.parse_args(remained_args)

  rendered_yaml = rendering_template(template_content, args)

  if rendered_yaml['path']['pretrain_network_g'] == None:
    import shutil
    rendered_yaml['path']['pretrain_network_g'] = f'experiments/{rendered_yaml["name"]}/models/net_g_latest.pth'
    copy_path = f'pretrained_models/temp/{rendered_yaml["name"]}.pth'
    shutil.copy(rendered_yaml['path']['pretrain_network_g'], copy_path)

  opt = parse(is_train=False, opt_content=rendered_yaml)
  opt['dist'] = False

  print(dict2str(opt))

  make_exp_dirs(opt)

  dataset_opt = opt['datasets']['val']
  dataset_name = dataset_opt['name']
  testset:SiddBenchmarkDataset = create_dataset(dataset_opt)

  visual_dir = osp.join(opt['path']['visualization'], dataset_name)
  if opt['val']['save_img'] and not os.path.exists(visual_dir):
    os.mkdir(visual_dir)
  print(f'total test samples: {len(testset)}')

  csvfile = open(osp.join(opt['path']['results_root'], f'{dataset_name}.csv'), 'w')
  writer = csv.writer(csvfile)
  writer.writerow(['patch_index', 'image_index', 'psnr', 'ssim', 'psnr_of_est', 'ssim_of_est', 'psnr_on_est', 'ssim_on_est'])

  dldns_model:ImageRestorationModel = create_model(opt)

  
  # torchinfo.summary(dldns_model.net_g, input_size=testset[0]['lq'].shape, verbose=2, depth=1)
  torchinfo.summary(dldns_model.net_g, input_size=testset[0]['lq'].shape)

  psnr_sum = 0
  ssim_sum = 0
  psnr_of_est_sum = 0
  ssim_of_est_sum = 0
  psnr_on_est_sum = 0
  ssim_on_est_sum = 0
  pbar = tqdm(total=len(testset), unit='image')
  for i in range(len(testset)):
    sample = testset[i]
    sample_id = Path(sample['lq_path']).stem.split('_')[-1]

    img_id = sample['meta_path'].stem.split('_')[0]

    dldns_model.feed_data(sample)
    dldns_model.test()
    dldns_out = dldns_model.get_current_visuals()
    dldns_out = dldns_out['result'].squeeze().numpy()
    # print(dldns_out.shape)

    isp_metadata_path = str(sample['meta_path'])
    isp_metadata = sio.loadmat(isp_metadata_path)
    isp_metadata = [(key, value) for key, value in zip( isp_metadata['metadata'][0].dtype.names, isp_metadata['metadata'][0][0])]
    isp_metadata = dict(isp_metadata)
    cfa_pattern = get_pattern_list(isp_metadata_path)
    pattern = get_pattern(isp_metadata_path)
    whitelevel = isp_metadata['AsShotNeutral'][0]
    color_matrix = isp_metadata['ColorMatrix2'].astype(np.float64)

    if wb_first:
      demosaic_in = np.clip(dldns_out, 0, 1)
    else:
      wb_out = white_balance(dldns_out, whitelevel, cfa_pattern)
      demosaic_in = np.clip(wb_out, 0, 1)

    demosaic_in = np.round(np.clip(demosaic_in*(2**16), 0, (2**16)-1)).astype(np.uint16)
    demosaic_out = demosaic_matlab(demosaic_in, pattern)
    demosaic_out = demosaic_out.astype(np.float32)/(2**16)

    xyz_image = apply_color_space_transform(demosaic_out, color_matrix_2=color_matrix)
    srgb_out = transform_xyz_to_srgb(xyz_image)

    result_int = np.round(srgb_out * 255).astype(np.uint8)
    gt_int = np.round(sample['gt'].squeeze().numpy() * 255).astype(np.uint8)
    # psnr = peak_signal_noise_ratio(gt_int, result_int, data_range=256)
    # ssim = structural_similarity(gt_int, result_int, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
    # ssim = structural_similarity(gt_int, result_int, data_range=256, channel_axis=2)
    psnr = calculate_psnr(result_int, gt_int, crop_border=0)
    ssim = calculate_ssim(result_int, gt_int, crop_border=0)

    #
    gt_raw = sample['gt_raw'].squeeze().numpy()
    # print(gt_raw.shape)
    if wb_first:
      demosaic_in = np.clip(gt_raw, 0, 1)
    else:
      wb_out = white_balance(gt_raw, whitelevel, cfa_pattern)
      demosaic_in = np.clip(wb_out, 0, 1)
    demosaic_in = np.round(np.clip(demosaic_in*(2**16), 0, (2**16)-1)).astype(np.uint16)
    demosaic_out = demosaic_matlab(demosaic_in, pattern)
    demosaic_out = demosaic_out.astype(np.float32)/(2**16)

    xyz_image = apply_color_space_transform(demosaic_out, color_matrix_2=color_matrix)
    srgb_out = transform_xyz_to_srgb(xyz_image)
    gt_estimate_int = np.round(srgb_out * 255).astype(np.uint8)
    # psnr_of_est = peak_signal_noise_ratio(gt_int, gt_estimate_int, data_range=256)
    # ssim_of_est = structural_similarity(gt_int, gt_estimate_int, data_range=256, channel_axis=2)
    # psnr_of_est = calculate_psnr(gt_int, gt_estimate_int, crop_border=0)
    # ssim_of_est = calculate_ssim(gt_int, gt_estimate_int, crop_border=0)
    psnr_of_est = 1
    ssim_of_est = 1

    # psnr_on_est = peak_signal_noise_ratio(gt_estimate_int, result_int, data_range=256)
    # ssim_on_est = structural_similarity(gt_estimate_int, result_int, data_range=256, channel_axis=2)
    # psnr_on_est = calculate_psnr(gt_estimate_int, result_int, crop_border=0)
    # ssim_on_est = calculate_ssim(gt_estimate_int, result_int, crop_border=0)
    psnr_on_est = 1
    ssim_on_est = 1

    if opt['val']['save_img']:
      cv2.imwrite(osp.join(visual_dir, f"{sample_id}_gt.png"), gt_int)
      cv2.imwrite(osp.join(visual_dir, f"{sample_id}_lq.png"), result_int)
      # cv2.imwrite(osp.join(visual_dir, f"{sample['meta_path'].stem}_gt.png"), gt_int)
      # cv2.imwrite(osp.join(visual_dir, f"{sample['meta_path'].stem}_lq.png"), result_int)

    psnr_sum += psnr
    ssim_sum += ssim
    psnr_of_est_sum += psnr_of_est
    ssim_of_est_sum += ssim_of_est
    psnr_on_est_sum += psnr_on_est
    ssim_on_est_sum += ssim_on_est

    # print(sample['meta_path'].stem, f"psnr: {psnr}, ssim: {ssim}")
    # print(f"{sample_id:04d}, psnr: {psnr:.3f}, ssim: {ssim:.3f}")
    writer.writerow([sample_id, img_id, psnr, ssim, psnr_of_est, ssim_of_est, psnr_on_est, ssim_on_est])
    pbar.update(1)

  n = len(testset)
  psnr_avg = psnr_sum / n
  ssim_avg = ssim_sum / n
  psnr_of_est_avg = psnr_of_est_sum / n
  ssim_of_est_avg = ssim_of_est_sum / n
  psnr_on_est_avg = psnr_on_est_sum / n
  ssim_on_est_avg = ssim_on_est_sum / n
  print(f"exp name: {opt['name']}")
  print(f"psnr: {psnr_avg:.3f}, ssim: {ssim_avg:.3f}")
  print(f"psnr (of est): {psnr_of_est_avg:.3f}, ssim (of est): {ssim_of_est_avg:.3f}")
  print(f"psnr (on est): {psnr_on_est_avg:.3f}, ssim (on est): {ssim_on_est_avg:.3f}")
  csvfile.close()


if __name__ == "__main__":
  wb_first=False
  main(wb_first)
