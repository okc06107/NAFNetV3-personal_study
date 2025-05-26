import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

from basicsr.utils.options import dict2str, parse
from basicsr.data import create_dataset

from basicsr.data.sidd_benchmark_dataset import SiddBenchmarkDataset

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

parser = ArgumentParser()
parser.add_argument('-opt', type=str, default='scripts/test/test_sidd_benchmark.yml')

args = parser.parse_args()
opt = parse(args.opt, is_train=False)

print(dict2str(opt))

dataset_opt = opt['datasets']['val']

testset:SiddBenchmarkDataset = create_dataset(dataset_opt)

sample = testset[0]
print(sample['gt_path']) # rgb
print(sample['lq_path']) # raw
print(sample['meta_path'])
print(type(sample['meta_path']))

bayer_in = sample['lq']
# temp = testset.preproc(bayer_in, str(sample['meta_path']))
rgb_out = testset.postproc(bayer_in, str(sample['meta_path']))

plt.imsave('gt_est.png', rgb_out)
plt.imsave('gt.png', sample['gt'])

# pipeline test
gt_int = np.round(sample['gt'] * 255).astype(np.uint8)
gt_estimate_int = np.round(rgb_out * 255).astype(np.uint8)


psnr_of_est = peak_signal_noise_ratio(gt_int, gt_estimate_int, data_range=256)
print(f"psnr (of gt estimate): {psnr_of_est:.3f}")
ssim_of_est = structural_similarity(gt_int, gt_estimate_int, data_range=256, channel_axis=2)
print(f"ssim (of gt estimate): {ssim_of_est:.3f}")