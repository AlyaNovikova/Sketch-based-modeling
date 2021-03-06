# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import wandb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    wandb.init(project='sketch')
    wandb_id = int(wandb.run.name[-3:])
    print('wandb id', wandb_id)
    final_output_dir += str(wandb_id)

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    writer_dict['writer'].add_graph(model, (dump_input,))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    bce = nn.BCEWithLogitsLoss()
    alpha = cfg.DISCRIMINATOR_COEFFICIENT

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + 'united')(
        cfg, False, cfg.DATASET.DATASET, cfg.DATASET.ROOT, False
    )
    print('Len of train_dataset', len(train_dataset))

    valid_dataset = eval('dataset.' + 'united')(
        cfg, True, cfg.DATASET.DATASET, cfg.DATASET.ROOT, False
    )
    print('Len of valid_dataset', len(valid_dataset))

    valid_drawings_dataset = eval('dataset.' + 'united')(
        cfg, True, cfg.DATASET.DRAWINGS_DATASET, cfg.DATASET.DRAWINGS_DATASET_ROOT, True
    )
    print('Len of valid_drawings_dataset', len(valid_drawings_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    eval_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    eval_drawings_loader = torch.utils.data.DataLoader(
        valid_drawings_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    valid_dataset_drawings = eval('dataset.' + cfg.DATASET.DRAWINGS_DATASET)(
        cfg, cfg.DATASET.DRAWINGS_DATASET_ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader_drawings = torch.utils.data.DataLoader(
        valid_dataset_drawings,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    def infinite_iterator(loader):
        while True:
            yield from loader

    eval_loader = infinite_iterator(eval_loader)
    eval_drawings_loader = infinite_iterator(eval_drawings_loader)

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    max_val_map = 0.0

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        max_val_map = train(cfg, train_loader, model, criterion, bce, alpha, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict,
              {'sketch_val': eval_loader, 'drawings_val': eval_drawings_loader},
              valid_loader_drawings, valid_dataset_drawings, max_val_map)

        # evaluate on validation set
        # perf_indicator = validate(
        #     cfg, valid_loader, valid_dataset.dataset, model, criterion,
        #     final_output_dir, tb_log_dir, writer_dict
        # )

        # if perf_indicator >= best_perf:
        #     best_perf = perf_indicator
        #     best_model = True
        # else:
        #     best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'model': cfg.MODEL.NAME,
        #     'state_dict': model.state_dict(),
        #     'best_state_dict': model.module.state_dict(),
        #     'perf': perf_indicator,
        #     'optimizer': optimizer.state_dict(),
        # }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
