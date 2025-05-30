import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str

def main():
  opt = parse_options(is_train=False)
  torch.backends.cudnn.benchmark = True

  make_exp_dirs(opt)
  log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
  logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
  logger.info(get_env_info())
  logger.info(dict2str(opt))

  test_loaders = []
  for phase, dataset_opt in sorted(opt['datasets'].items()):
    if 'test' in phase:
      dataset_opt['phase'] = 'test'
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(
      test_set,
      dataset_opt,
      num_gpu=opt['num_gpu'],
      dist=opt['dist'],
      sampler=None,
      seed=opt['manual_seed']
    )
    logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
    test_loaders.append(test_loader)

  model = create_model(opt)
  for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info(f'Testing {test_set_name}...')

if __name__ == "__main__":
  main()