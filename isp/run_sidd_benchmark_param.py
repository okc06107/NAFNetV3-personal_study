import os
import csv
import numpy as np
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

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from basicsr.metrics import calculate_psnr, calculate_ssim

from basicsr.train_param import make_yaml_parser, rendering_template
import torchinfo

def main():
    parser = ArgumentParser()
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
    writer.writerow(['patch_index', 'psnr', 'ssim'])

    dldns_model:ImageRestorationModel = create_model(opt)
    
    # Get input shape from first sample and ensure batch size is 1
    input_shape = testset[0]['lq'].shape
    if len(input_shape) == 3:  # If input is (C, H, W)
        input_shape = (1,) + input_shape  # Add batch dimension: (1, C, H, W)
    elif len(input_shape) == 4 and input_shape[0] != 1:  # If batch size is not 1
        input_shape = (1,) + input_shape[1:]  # Set batch size to 1
    
    with open("summary.txt", "w") as f:
        f.write(str(torchinfo.summary(dldns_model.net_g, input_size=input_shape)))

    psnr_sum = 0
    ssim_sum = 0
    pbar = tqdm(total=len(testset), unit='image')
    for i in range(len(testset)):
        sample = testset[i]
        sample_id = Path(sample['lq_path']).stem.split('_')[-1]

        dldns_model.feed_data(sample)
        dldns_model.test()
        dldns_out = dldns_model.get_current_visuals()
        result = dldns_out['result'].squeeze().numpy()

        result_int = np.round(result * 255).astype(np.uint8)
        gt_int = np.round(sample['gt'].squeeze().numpy() * 255).astype(np.uint8)

        psnr = calculate_psnr(result_int, gt_int, crop_border=0)
        ssim = calculate_ssim(result_int, gt_int, crop_border=0)

        if opt['val']['save_img']:
            cv2.imwrite(osp.join(visual_dir, f"{sample_id}_gt.png"), gt_int)
            cv2.imwrite(osp.join(visual_dir, f"{sample_id}_lq.png"), result_int)

        psnr_sum += psnr
        ssim_sum += ssim

        writer.writerow([sample_id, psnr, ssim])
        pbar.update(1)

    n = len(testset)
    psnr_avg = psnr_sum / n
    ssim_avg = ssim_sum / n
    print(f"exp name: {opt['name']}")
    print(f"psnr: {psnr_avg:.3f}, ssim: {ssim_avg:.3f}")
    csvfile.close()

if __name__ == "__main__":
    main()
