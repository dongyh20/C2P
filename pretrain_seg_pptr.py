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

from datasets.seg_pretrain import SegDataset
import models.seg_pptr_pretrain as Models  #################


def train_one_epoch(model, forward_predictor, back_predictor, criterion, optimizer, lr_scheduler, data_loader, device,
                    epoch, print_freq):
    model.train()
    forward_predictor.train()
    back_predictor.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for clip, clip_comp in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        clip, clip_comp = clip.to(device), clip_comp.to(device)
        rgb = torch.swapaxes(clip, 2, 3)
        rgb_comp = torch.swapaxes(clip_comp, 2, 3)
        output = model(clip, rgb)[1]
        output_comp = model(clip_comp, rgb_comp)[0]

        B, L, C = output.shape

        loss = 0

        loss_distill = 0
        for b in range(B):
            out = output[b]

            out_comp = output_comp[b]

            out = torch.nn.functional.normalize(out, dim=1)
            out_comp = torch.nn.functional.normalize(out_comp, dim=1)

            logits = torch.mm(out, out_comp.transpose(1, 0))

            labels = torch.arange(out.size()[0])
            labels = labels.cuda()
            loss_tmp = criterion(logits / 0.07, labels)

            loss_distill = loss_tmp + loss_distill

        loss_distill = loss_distill / B

        loss_predict = 0
        for b in range(B):
            anchor_feature = output[b]
            comp_feature = output_comp[b]

            predict_feature = forward_predictor(anchor_feature[:-1])

            predict_feature_new = predict_feature
            key_feature_new = comp_feature[1:]

            predict_feature = torch.nn.functional.normalize(predict_feature_new, dim=1)
            key_feature = torch.nn.functional.normalize(key_feature_new, dim=1)

            logits = torch.mm(predict_feature, key_feature.transpose(1, 0))

            labels = torch.arange(key_feature.size()[0])
            labels = labels.cuda()
            loss_tmp = criterion(logits / 0.07, labels)

            loss_predict = loss_tmp + loss_predict

        loss_predict = loss_predict / B

        loss_back_predict = 0
        for b in range(B):
            anchor_feature = output[b]
            comp_feature = output_comp[b]

            predict_feature = back_predictor(anchor_feature[1:])

            predict_feature_new = predict_feature
            key_feature_new = comp_feature[:-1]

            predict_feature = torch.nn.functional.normalize(predict_feature_new, dim=1)
            key_feature = torch.nn.functional.normalize(key_feature_new, dim=1)

            logits = torch.mm(predict_feature, key_feature.transpose(1, 0))

            labels = torch.arange(key_feature.size()[0])
            labels = labels.cuda()
            loss_tmp = criterion(logits / 0.07, labels)

            loss_back_predict = loss_tmp + loss_back_predict

        loss_back_predict = loss_back_predict / B

        loss = loss_distill * 0.5 + (loss_predict + loss_back_predict) / 2 * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = out.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()


def main(args):
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

    dataset = SegDataset(root='/datasets/Seg_pre', root_complete='/datasets/Seg_data',
                         meta='/release.txt', frames_per_clip=10, num_points=4096, train=True)
    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True, drop_last=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples)
    MLP = getattr(Models, 'MLP')
    forward_predictor = MLP(
        1024, 1024
    )
    back_predictor = MLP(
        1024, 1024
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        forward_predictor = nn.DataParallel(forward_predictor)
        back_predictor = nn.DataParallel(back_predictor)
    model.to(device)
    forward_predictor.to(device)
    back_predictor.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(forward_predictor.parameters()) + list(back_predictor.parameters()), lr=lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
                                     warmup_iters=warmup_iters, warmup_factor=1e-5)

    # model_without_ddp = model

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, forward_predictor, back_predictor, criterion, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq)

        if (epoch + 1) % 1 == 0:
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
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='PrimitiveTransformer', type=str, help='model')
    # P4D
    parser.add_argument('--radius', default=0.9, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=1, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=2048, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=1024, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=20, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[12, 17], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.5, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str,
                        help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
