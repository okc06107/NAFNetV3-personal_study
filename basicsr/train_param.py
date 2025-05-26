import argparse
import datetime
import logging
import math
import random
import time
import torch
import re
import yaml
from os import path as osp

import torch.distributed

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import jinja2 as j2
import torchinfo

def remove_leading_underscore(s):
    if s.startswith("_"):
        return s[1:]
    else:
        return s

def make_yaml_parser(template_content):
    parser = argparse.ArgumentParser()
    matches_with_type = re.findall(r"!!(\w+)\s*{{\s*(\w+)\s*}}", template_content)
    matches_without_type = re.findall(r"{{\s*(\w+)\s*}}", template_content)

    type_mapping = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool
    }

    for var_type, var_name in matches_with_type:
        arg_type = type_mapping.get(var_type, str)
        if var_type == 'bool':
            parser.add_argument(f"--{remove_leading_underscore(var_name)}", action="store_true")
        else:
            parser.add_argument(f"--{remove_leading_underscore(var_name)}", type=arg_type, required=True)
    
    for var_name in matches_without_type:
        if var_name not in [name for _, name in matches_with_type]:
            arg_type = int
            parser.add_argument(f"--{remove_leading_underscore(var_name)}", type=arg_type, required=True)

    return parser

def rendering_template(template_content, args):
    _variables = re.findall(r"{{\s*(\w+)\s*}}", template_content)
    variables = list(set(_variables))
    # 원하는 순서로 변수명 매핑
    custom_order = ['ch_mid', 'end_type', 'epoch', 'mid', 'mid_type', 'shuffle1', 'width']
    # 변수명 매핑: ch_mid->chmid, end_type->endtype, mid_type->midtype
    varname_map = {'ch_mid': 'chmid', 'end_type': 'endtype', 'mid_type': 'midtype'}
    name_postpix_parts = []
    for k in custom_order:
        if k in variables:
            v = getattr(args, remove_leading_underscore(k))
            mapped = varname_map.get(k, k)
            name_postpix_parts.append(f"{mapped}{v}")
    name_postpix = "_".join(name_postpix_parts)
    variables = {var: getattr(args, remove_leading_underscore(var)) for var in variables}
    template = j2.Template(template_content)
    rendered_content = template.render(variables)
    config = yaml.safe_load(rendered_content)
    config['name'] = config['name']+"_"+name_postpix
    return config


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--template', help="Path to option YAML template")
    args, remained_args = parser.parse_known_args()

    template_path = args.template
    with open(template_path, 'r') as file:
        template_content = file.read()
    parser = make_yaml_parser(template_content)

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args(remained_args)
    # args.opt = rendering_template(template_path, template_content, args)
    rendered_yaml = rendering_template(template_content, args)

    # print(rendered_yaml['name'])
    # opt = parse(args.opt, is_train=is_train)
    opt = parse(is_train=is_train, opt_content=rendered_yaml)

    # print(dict2str(opt))
    
    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        opt_folder = osp.dirname(template_path)
        opt_file_name = opt['name']+".yml"
        opt_path = osp.join(opt_folder, opt_file_name)
        with open(opt_path, 'w') as file:
            yaml.dump(rendered_yaml, file, sort_keys=False)

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=f'./logs/{opt['name']}') #mkdir logs @CLY
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    model_folder_path = 'experiments/{}/models/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        print('!!!!!! resume state .. ', states, state_folder_path)
        # 최신 1개의 state만 유지
        state_files = sorted([int(x[0:-6]) for x in states])
        if len(state_files) > 1:
            for state_file in state_files[:-1]:
                os.remove(os.path.join(state_folder_path, f'{state_file}.state'))
        max_state_file = '{}.state'.format(max(state_files))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        print("resume_state is None")
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)
    # tb_logger: SummaryWriter(log_dir=log_dir)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
    

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        if current_iter == 0 and opt['rank'] == 0:
            torchinfo.summary(model.net_g, input_size=train_data['lq'].shape, depth=4)

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            result_code = model.optimize_parameters(current_iter, tb_logger)
            # if result_code == -1 and tb_logger:
            #     print('loss explode .. ')
            #     exit(0)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                # print('msg logger .. ', current_iter)
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving training states.')
                # 최근 1개의 training state만 유지
                try:
                    state_files = [f for f in os.listdir(state_folder_path) if f.endswith('.state')]
                    state_iters = sorted([int(f[:-6]) for f in state_files if f[:-6].isdigit()])
                    if len(state_iters) > 1:
                        for iter_num in state_iters[:-1]:
                            old_state_path = os.path.join(state_folder_path, f'{iter_num}.state')
                            if os.path.exists(old_state_path):
                                os.remove(old_state_path)
                except:
                    pass
                # 현재 training state 저장
                model.save(epoch, current_iter)
                # === 중간 체크포인트는 1개만 유지 ===
                try:
                    for f in os.listdir(model_folder_path):
                        if f.startswith('net_g_') and f.endswith('.pth') and f != f'net_g_{current_iter}.pth':
                            os.remove(os.path.join(model_folder_path, f))
                except Exception as e:
                    logger.warning(f"Failed to delete old checkpoints: {e}")

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0 or current_iter == 1000):
            # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)


            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    # === net_g_latest.pth만 남기고 모두 삭제 ===
    try:
        for f in os.listdir(model_folder_path):
            if f.startswith('net_g_') and f.endswith('.pth') and f != 'net_g_latest.pth':
                os.remove(os.path.join(model_folder_path, f))
    except Exception as e:
        logger.warning(f"Failed to delete old checkpoints after latest: {e}")

    if opt.get('val') is not None:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        metric = model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'], rgb2bgr, use_image)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
