# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import wandb

from core.evaluate import accuracy
from core.inference import get_final_preds, get_max_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images, save_debug_images_inference

logger = logging.getLogger(__name__)


def train_step(step, model, input, target, target_weight, meta, domain, criterion, bce, alpha, optimizer):
    # compute output
    loss, avg_acc, cnt, pred, output = forward(
        alpha, bce, criterion, domain, input, model, target, target_weight, 'train', step
    )

    # compute gradient and do update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, avg_acc, cnt, pred, output


def forward(alpha, bce, criterion, domain, input, model, target, target_weight, tag, step):
    output, d_pred = model(input)
    domain = domain.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    target_weight = target_weight.cuda(non_blocking=True)
    if sum(domain == 1) == 0:
        loss1 = 0
    else:
        loss1 = criterion(output[domain == 1], target[domain == 1], target_weight[domain == 1])
    # loss1 = criterion(output[domain == 1], target[domain == 1], target_weight[domain == 1])
    loss2 = bce(d_pred, domain.float()* .9 + 0.05)
    loss = loss1 + alpha * loss2

    _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                     target.detach().cpu().numpy())

    for d_number in [0, 1]:
        d_indicator = domain == d_number
        if d_indicator.sum() > 0:
            d_acc = (domain[d_indicator] == (d_pred[d_indicator] > 0)).float().mean()
            wandb.log(
                {
                    f'{tag}/disc_acc_{d_number}': d_acc,
                    f'{tag}/disc_mean_logit_{d_number}': d_pred[d_indicator].mean()
                },
                step=step
            )

    wandb.log(
        {
            f'{tag}/pose_loss': loss1,
            f'{tag}/discr_loss': loss2,
            f'{tag}/total_loss': loss,
            f'{tag}/pose_acc': avg_acc,
        },
        step=step,
    )

    return loss, avg_acc, cnt, pred, output


def eval_step(step, model, eval_loader, criterion, bce, alpha, plot_name):
    model.eval()

    with torch.no_grad():
        input, target, target_weight, meta, domain = next(eval_loader)
        forward(alpha, bce, criterion, domain, input, model, target, target_weight, plot_name, step)

    model.train()


def train(config, train_loader, model, criterion, bce, alpha, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, eval_loaders):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, target_weight, meta, domain) in enumerate(train_loader):
        data_time.update(time.time() - end)

        loss, avg_acc, cnt, pred, output = train_step(
            epoch * len(train_loader) + i, model, input, target, target_weight, meta, domain, criterion, bce, alpha, optimizer
        )

        losses.update(loss.item(), input.size(0))
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.EVAL_FREQ == 0:
            for plot_name, val_loader in eval_loaders.items():
                eval_step(epoch * len(train_loader) + i, model, val_loader, criterion, bce, alpha, plot_name)

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred * 4, output, prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        # print(len(val_loader))
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # print('?????????????????')
            # print(i)
            # print(input)
            # compute output
            outputs, d_pred = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            #             if config.TEST.FLIP_TEST:
            #                 input_flipped = input.flip(3)
            #                 outputs_flipped = model(input_flipped)

            #                 if isinstance(outputs_flipped, list):
            #                     output_flipped = outputs_flipped[-1]
            #                 else:
            #                     output_flipped = outputs_flipped

            #                 output_flipped = flip_back(output_flipped.cpu().numpy(),
            #                                            val_dataset.flip_pairs)
            #                 output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            #                 # feature is not aligned, shift flipped heatmap for higher accuracy
            #                 if config.TEST.SHIFT_HEATMAP:
            #                     output_flipped[:, :, :, 1:] = \
            #                         output_flipped.clone()[:, :, :, 0:-1]

            #                 output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            # if sum(domain == 1) == 0:
            #     loss1 = 0
            # else:
            #     loss1 = criterion(output[domain == 1], target[domain == 1], target_weight[domain == 1])
            # loss2 = bce(output, target)
            # loss = loss1 + alpha * loss2
            # wandb.log(
            #     {'val/pose_loss': loss1, 'val/discriminator': loss2, 'val/sum_loss': loss},
            #     step=epoch * len(val_loader) + i
            # )

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            # wandb.log(
            #     {'train/acc': avg_acc},
            #     step=epoch * len(val_loader) + i
            # )
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, output, prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def inference(config, val_loader, val_dataset, model, criterion, output_dir,
              tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            num_images = input.size(0)
            pred, _ = get_max_preds(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            #             print(pred)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'inference_val'), i
                )
                save_debug_images_inference(config, input, meta, target, pred * 4, output,
                                            prefix)

        model_name = config.MODEL.NAME

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer_dict['valid_global_steps'] = global_steps + 1

    return 0


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
