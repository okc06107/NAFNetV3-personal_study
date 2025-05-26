import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path

from basicsr.utils.options import dict2str, parse
from basicsr.data import create_dataset

parser = ArgumentParser()
parser.add_argument('-opt', type=str, default='scripts/test/test_sidd_raw.yml')

args = parser.parse_args()
opt = parse(args.opt, is_train=True)

print(dict2str(opt))

for phase, dataset_opt in opt['datasets'].items():
  dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
  train_set = create_dataset(dataset_opt)
  pass

sample = train_set[0]
# sample_lq = sample['lq']
print(sample['gt'].shape)
print(sample['gt'].dtype)
img_gt = np.round(sample['gt'].numpy() * 255).astype(np.uint8)
img_lq = np.round(sample['lq'].numpy() * 255).astype(np.uint8)
plt.imsave(Path(args.opt).parents[0].joinpath('test_sidd_raw_gt.png'), img_gt[0])
plt.imsave(Path(args.opt).parents[0].joinpath('test_sidd_raw_lq.png'), img_lq[0])