from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR

from datasets.seg_base import SegDataset
import models.seg_p4_base as Models


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    cnt = 0
    for pc1, rgb1, label1 in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()

        pc1, rgb1, label1 = pc1.to(device), rgb1.to(device), label1.to(device)
        output1 = model(pc1, rgb1).transpose(1, 2)
        loss1 = criterion(output1, label1)
        loss1 = torch.mean(loss1)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        metric_logger.update(loss=loss1.item(), lr=optimizer.param_groups[0]["lr"])
        lr_scheduler.step()
        sys.stdout.flush()


def main(args):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = SegDataset(root='/datasets/Seg_data', frames_per_clip=3, train=True)

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=False)

    print("Creating model")

    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, num_classes=49)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion_train = nn.CrossEntropyLoss(reduction='none')

    lr = args.lr
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    print("Start training")
    best_iou = 0
    start_time = time.time()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        pre_state = checkpoint['model']
        # for name in pre_state.keys():
        #     print(name)
        update_dict = {k: v for k, v in pre_state.items() if
                       k.startswith("module.conv") or k.startswith("module.transformer")}
        for name in update_dict.keys():
            print(name)
        net_state_dict = model.state_dict()
        # for name in net_state_dict.keys():
        #     print(name)
        net_state_dict.update(update_dict)
        model.load_state_dict(net_state_dict)
        # print(pre_state['pos_embedding.weight'])
        # print(model.state_dict()['pos_embedding.weight'])

    for epoch in range(args.start_epoch, args.epochs):
        # print("training start!")
        train_one_epoch(model, criterion_train, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq,
                        args)
        if (epoch + 1) % 2 == 0:
            if args.output_dir:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Transformer Model Training')

    parser.add_argument('--data-path', default='', help='data path')
    parser.add_argument('--label-weight', default='', help='training label weights')

    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=3, type=int, metavar='N', help='number of frames per clip')  ##############
    parser.add_argument('--num-points', default=8192, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.9, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    # parser.add_argument('--spatial-stride', default=16, type=int, help='spatial subsampling rate')
    # parser.add_argument('--temporal-kernel-size', default=1, type=int, help='temporal kernel size')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=2, type=int, help='transformer depth')
    parser.add_argument('--head', default=4, type=int, help='transformer head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=24, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')  # 0.01
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[15, 25, 35], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.4, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
