import sys
print(sys.path)
from argparse import ArgumentParser

from basicsr.utils.options import dict2str, parse
from basicsr.data import create_dataset

parser = ArgumentParser()
# parser.add_argument('-opt', type=str, default='options/train/SIDD/NAFNet-width32.yml') # SIDD original
parser.add_argument('-opt', type=str, default='options/train/SIDD/NAFNetv2.yml') # SIDD raw

args = parser.parse_args()
opt = parse(args.opt, is_train=True)

for phase, dataset_opt in opt['datasets'].items():
  dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
  train_set = create_dataset(dataset_opt)
  pass

sample = train_set[0]
print(sample['gt'].shape)
print(sample['gt'].dtype)
print(sample['gt'].numpy())