#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import argparse
import copy
import random
import socket
from contextlib import suppress
from functools import partial
from tqdm import tqdm
import torch.nn as nn
import numpy as np

import time
import torch.distributed
import torch.distributed as dist
import torch.utils.data
from timm.data import Mixup
from timm.data.dataset import ImageDataset
from timm.loss import SoftTargetCrossEntropy
from timm.utils import random_seed, NativeScaler, accuracy
from timm.models import create_model, safe_model_name
from torch.amp import autocast as amp_autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.scheduler.scheduler_factory import CosineLRScheduler
from torch.utils.data import Dataset

from models import quant_swin_transformer, swin_transformer

import pytorch_quantization.nn as quant_nn

from utils.utils import write, create_transform, create_loader, AverageMeter, broadcast_tensor_from_main_process, gather_tensor_from_multi_processes, compute_quantized_params, collect_stats_format, compute_amax_format

HOST_NAME = socket.getfqdn(socket.gethostname())


torch.backends.cudnn.benchmark = True

#使用512张图片计算W和b
LINEAR_COMPENSATION_SAMPLES = 512


def mark_trainable_parameters(model: nn.Module, model_type):
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    try:
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True
        if hasattr(model, 'head_dist') and model.head_dist is not None:
            model.head_dist.weight.requires_grad = True
            model.head_dist.bias.requires_grad = True
    except:
        model.module.head.weight.requires_grad = True
        model.module.head.bias.requires_grad = True
        if hasattr(model.module, 'head_dist') and model.module.head_dist is not None:
            model.module.head_dist.weight.requires_grad = True
            model.module.head_dist.bias.requires_grad = True

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


#这个类对vit里面的每一个block进行封装，将block作为该类的一个成员变量，并添加W和b的初始化以及计算的逻辑
class CompensationBlock(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True, local_rank=0, block_id=None):
        super(CompensationBlock, self).__init__()
        self.block = block

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))

        #这里当r2_score大于0的时候才对W和b进行补偿，实验发现如果r2_score很小(<0)，初始化之后会掉很多个点
        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            if local_rank == 0:
                _write('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                _write('block {} using lora init'.format(block_id))

    def forward(self, x):
        out = self.block(x)
        out = out + x @ self.lora_weight + self.lora_bias

        return out

############################################
############################################
#如下的两个函数负责开启vit中每一个block的量化与关闭量化(fp32模式运行)的过程，不同的ptq方法是不一样的，所以需要针对不同的ptq方法进行修改
def enable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_quant()


def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_quant()
############################################
############################################


class FeatureDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]

#线性回归函数，根据输入X和Y计算W和b的解析解
def lienar_regression(X, Y, block_id=0):
    X = X.reshape(-1, X.size(-1))

    #同步不同进程的X和Y，确保不同进程的X和Y是一样的，不然在多进程的情况下算出来的X和Y不够512个样本
    X = gather_tensor_from_multi_processes(X, world_size=args.world_size)

    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))

    #同步不同进程的X和Y，确保不同进程的X和Y是一样的，不然在多进程的情况下算出来的X和Y不够512个样本
    Y = gather_tensor_from_multi_processes(Y, world_size=args.world_size)

    _write('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))

    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b

    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    _write('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))

    #输出W,b，以及r2_score, r2_score用来当作一个block是否启用初始化(或0初始化)的flag变量
    return W, b, r2_score

@torch.no_grad()
def generate_compensation_model(q_model, train_loader, args):
    _write('start to generate compensation model')

    #下面的代码块将图片转换为网络self.blocks层前的特征
    torch.cuda.synchronize()
    output_t = torch.zeros(size=[0,], device=args.device)
    for i, (image, _) in tqdm(enumerate(train_loader)):
        image = image.cuda()
        t_out = q_model.forward_before_blocks(image)
        output_t = torch.cat([output_t, t_out.detach()], dim=0)
        torch.cuda.synchronize()

        #如果够512张图像了就break
        if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
            break


    #将blocks之前的特征组织在一个FeatureDataset里面，这样当不同block的X计算完成之后，改feature_ser.X就行，非常方便
    feature_set = FeatureDataset(output_t.detach().cpu())
    feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_previous = output_t
    for layer_id in range(len(q_model.layers)):
        current_layer = q_model.layers[layer_id]
        for block_id in range(len(current_layer.blocks)):

            #把前一个block补偿之后的输出作为 当前block计算W和b的输入，第0个block则将forward_before_blocks(self.blocks)前的特征作为输入
            feature_set.X = output_previous.detach().cpu()

            block = current_layer.blocks[block_id]
            output_full_precision = torch.zeros(size=[0, ], device=args.device)
            output_quant = torch.zeros(size=[0, ], device=args.device)
            output_t_ = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()

                #disable quant,得到当前block fp32的输出
                disable_quant(block)
                full_precision_out = block(t_out)

                #enable quant,得到当前block quant的输出
                enable_quant(block)
                quant_out = block(t_out)

                #拼接各种变量
                output_t_ = torch.cat([output_t_, t_out.detach()], dim=0)
                output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
                output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size  // args.world_size - 1):
                    break

            assert torch.sum((output_previous - output_t_).abs()) < 1e-3
            global_block_id = sum(q_model.depths[:layer_id]) + block_id

            #线性回归计算W,b并得到r2_score
            W, b, r2_score = lienar_regression(output_t_, output_full_precision - output_quant, block_id=global_block_id)

            #将当前层与W,b封装在一个新的CompensationBlock对象中
            current_layer.blocks[block_id] = CompensationBlock(W=W, b=b, r2_score=r2_score, block=current_layer.blocks[block_id], linear_init=True if global_block_id >= args.start_block else False, local_rank=args.local_rank, block_id=global_block_id)
            q_model.cuda()


            qwerty_block = current_layer.blocks[block_id]

            #得到当前block经过W和b补偿之后的输出，并将这个输出在下一个block循环开始时作为其输入
            output_previous = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                enable_quant(qwerty_block)
                previous_out = qwerty_block(t_out)

                if (current_layer.downsample is not None) and (block_id == len(current_layer.blocks)-1):
                    previous_out = current_layer.downsample(previous_out)

                output_previous = torch.cat([output_previous, previous_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
                    break

    return q_model


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="swin_small", choices=['vit_small', 'vit_base', 'deit_tiny', 'deit_small', 'deit_base', 'swin_tiny', 'swin_small',  'deit_tiny_distilled', 'deit_small_distilled', 'deit_base_distilled'], help="model")
parser.add_argument('--dataset', default="imagenet")

parser.add_argument('--num_bits', default=6, type=int, help='bit-precision of weights')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--start_block', default=0, type=int)

parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument("--batch_size", default=32, type=int, help="batchsize of validation set")
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 1e-4)')
parser.add_argument('--lr', type=float, default=5e-7, metavar='LR', help='learning rate (default: 1e-3)')
parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.2)')
parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')

parser.add_argument('--inner_eval', default=True, action='store_true')
parser.add_argument('--save_files', default=True, action='store_true')
parser.add_argument('--amp', default=True, action='store_true')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--seed", default=0, type=int, help="seed")

if '3090' in HOST_NAME:
    parser.add_argument("--local-rank", default=0, type=int)
else:
    parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

eval_period = 1
train_aug = 'large_scale_train'
test_aug = 'large_scale_test'
if 'admin' in HOST_NAME:
    args.data_dir = '/opt/Dataset/ImageNet'
else:
    args.data_dir = '/mnt/ramdisk/fumh/datasets/ImageNet'
args.num_classes = 1000

model_type = args.model.split("_")[0]
if model_type == "deit":
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_pct = 0.875
elif model_type == 'vit':
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    crop_pct = 0.9
elif model_type == 'swin':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_pct = 0.9
else:
    raise NotImplementedError

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
args.device = 'cuda:0'
args.world_size = 1
args.rank = 0  # global rank
if args.distributed:
    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

assert args.rank >= 0


args.log_file = None

torch.cuda.synchronize()

_write = partial(write, log_file=args.log_file)

if args.distributed:
    _write('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
else:
    _write('Training with a single process on 1 GPUs.')
assert args.rank >= 0


def main():

    if args.local_rank == 0:
        _write(args)

    seed(args.seed)


    if args.local_rank == 0:
        _write('dataset mean : {} & std : {}'.format(mean, std))

    if args.dataset == 'imagenet':
        dataset_train = ImageDataset(root=os.path.join(args.data_dir, 'train'), transform=create_transform(train_aug, mean, std, crop_pct))
        dataset_eval = ImageDataset(root=os.path.join(args.data_dir, 'val'), transform=create_transform(test_aug, mean, std, crop_pct))
    else:
        raise NotImplementedError

    if args.local_rank == 0:
        _write('len of train_set : {}    train_transform : {}'.format(len(dataset_train), dataset_train.transform))
        _write('len of eval_set : {}    eval_transform : {}'.format(len(dataset_eval), dataset_eval.transform))

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.
    if args.local_rank == 0:
        _write('mixup_active : {}'.format(mixup_active))

    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    if args.local_rank == 0:
        _write('collate_fn : {}'.format(collate_fn))
        _write('mixup_fn : {}'.format(mixup_fn))

    loader_train = create_loader(
        dataset_train,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=0.0,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=True,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    loader_eval = create_loader(
        dataset_eval,
        batch_size=args.batch_size,
        is_training=False,
        re_prob=0.,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=False,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    for data, target in loader_train:
        calib_data = data.to(args.device)
        break

    broadcast_tensor_from_main_process(calib_data, args)
    _write('local_rank : {} calib_data shape : {} value : {}'.format(args.local_rank, calib_data.size(), calib_data[0, 0, 0, :5]))

    if mixup_active:
        criterion = SoftTargetCrossEntropy()
        if args.local_rank == 0:
            _write('Using SoftTargetCrossEntropy')
    else:
        criterion = nn.CrossEntropyLoss()
        if args.local_rank == 0:
            _write('Using CrossEntropyLoss')

    loss_scaler = NativeScaler() if args.amp else None
    autocast = amp_autocast if args.amp else suppress

    if (loss_scaler is not None) and (autocast == amp_autocast):
        if args.local_rank == 0:
            _write('Training in AMP')
    else:
        if args.local_rank == 0:
            _write('Training in FP32')

    base_model_zoo = {
        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
    }

    model_zoo = {
        'swin_tiny' : 'swin_tiny_patch4_window7_224_quant',
        'swin_small': 'swin_small_patch4_window7_224_quant',
    }


    ###############################################
    #这一部分的代码是ptq模型初始化的代码，不同的ptq方法初始化的过程不一样，因此需要根据使用的ptq方法进行修改
    base_model = create_model(base_model_zoo[args.model], num_classes=args.num_classes, pretrained=True)

    base_model.cuda()
    base_model.eval()

    q_model = create_model(model_zoo[args.model], num_classes=args.num_classes, pretrained=True, num_bits=args.num_bits, log_file=args.log_file, drop_path_rate=args.drop_path)

    q_model.cuda()
    q_model.eval()

    with torch.no_grad():
        assert calib_data.size(0) == 32
        collect_stats_format(q_model, calib_data)
        compute_amax_format(q_model, calib_method='percentile')

    q_model.cuda()
    ##################################################

    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('ptq model size is {:.3f}'.format(ptq_params))

    top1_acc_eval = validate(q_model, loader_eval, autocast=autocast)
    _write('ptq   eval_acc: {:.2f}'.format(top1_acc_eval.avg))


    #调用函数生成不同block的W和b
    q_model = generate_compensation_model(q_model, loader_train, args)

    qwerty_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('qwerty model size is {:.3f}'.format(qwerty_params))

    if args.local_rank == 0:
        for n, m in q_model.named_modules():
            if isinstance(m, quant_nn.TensorQuantizer):
                _write('quant module : {}, calibrator : {}, enable_quant : {}, bits : {}'.format(n, m._calibrator, m._if_quant, m._num_bits))
                assert m._if_quant == True
                assert m._if_calib == False

    top1_acc_eval = validate(q_model, loader_eval, autocast=autocast)
    _write('compensation   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    # 如果不finetune的话下面的代码都可以不看了


#     if args.local_rank == 0:
#         for n, p in q_model.named_parameters():
#             if p.requires_grad:
#                 _write('requires_grad : {}  with shape {} | params : {}'.format(n, p.size(), p[0, :5].data if p.dim() == 2 else p[:5].data))
#
#     if args.eval:
#         return
#
#     if args.distributed:
#         q_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(q_model)
#         if args.local_rank == 0:
#             _write('Converted q_model to use Synchronized BatchNorm. WARNING: You may have issues if using ''zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
#             _write("Using native Torch DistributedDataParallel.")
#
#         q_model = NativeDDP(q_model, device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)
#
#
#     optimizer = torch.optim.AdamW(params=q_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#     if args.local_rank == 0:
#         _write(f'Model {args.model} created, param count:{sum([m.numel() for m in q_model.parameters()])}')
#
#         _write(f"number of params for requires grad in backbone (not head): {sum(p.numel() for n, p in q_model.named_parameters() if ((p.requires_grad) and ('head' not in n) and ('classifier' not in n) and ('fc.' not in n)))}")
#         _write(f"number of params for requires grad in backbone (count lora): {sum(p.numel() for n, p in q_model.named_parameters() if (p.requires_grad and ('lora_' in n)))}")
#
#     # lr_scheduler = CosineLRScheduler(
#     #     optimizer,
#     #     t_initial=args.epochs,
#     #     lr_min=args.lrmin,
#     #     warmup_lr_init=1e-7,
#     #     warmup_t=args.warmup_epochs)
#
#     num_epochs = args.epochs + args.warmup_epochs
#     if args.local_rank == 0:
#         _write('Scheduled epochs: {}'.format(num_epochs))
#
#     base_model.cuda()
#     best_acc = 0.0
#
#     start = time.time()
#     for epoch in range(1, num_epochs + 1):
#         if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
#             loader_train.sampler.set_epoch(epoch)
#
#         mark_trainable_parameters(q_model, args.model)
#         freeze_all_params(base_model)
#
#         train_one_epoch(epoch, base_model, q_model, loader_train, optimizer, criterion, args, autocast=autocast, loss_scaler=loss_scaler, mixup_fn=mixup_fn, start=start, num_epochs=num_epochs)
#         # lr_scheduler.step(epoch)
#
#         if args.inner_eval:
#             if epoch % eval_period == 0:
#                 top1_acc_eval = validate(q_model, loader_eval, autocast=autocast)
#
#                 _write('epoch: {}   eval_acc: {:.2f}'.format(epoch, top1_acc_eval.avg))
#
#                 if top1_acc_eval.avg > best_acc:
#                     best_acc = top1_acc_eval.avg
#                     _write('best acc {} in epoch {}, save files'.format(best_acc, epoch))
#
#                     if args.save_files:
#                         # torch.save(q_model.state_dict(), os.path.join(args.log_dir, 'best.pth'))
#                         if args.num_bits == 8:
#                             onnx_export(q_model, torch.randn(64, 3, 224, 224, device='cuda'), 'best')
#
#         torch.cuda.synchronize()
#
#     top1_acc_eval = validate(q_model, loader_eval, autocast=autocast)
#     _write('epoch: {}   eval_acc: {:.2f}'.format(epoch, top1_acc_eval.avg))
#
#     if args.save_files:
#         # torch.save(q_model.state_dict(), os.path.join(args.log_dir, 'last.pth'))
#         if args.num_bits == 8:
#             onnx_export(q_model, torch.randn(64, 3, 224, 224, device='cuda'), 'last')
#
#
# def train_one_epoch(epoch, base_model, model, loader, optimizer, loss_fn, args, autocast, loss_scaler=None, mixup_fn=None, start=None, num_epochs=None):
#     losses_m_cls = AverageMeter()
#     losses_m_mse = AverageMeter()
#
#     base_model.train()
#     model.train()
#
#     for batch_idx, (input, target) in enumerate(loader):
#
#         input, target = input.cuda(), target.cuda()
#         if mixup_fn is not None:
#             input, target = mixup_fn(input, target)
#
#         with autocast('cuda'):
#             with torch.no_grad():
#                 base_feat, _ = base_model(input)
#             feat, output = model(input)
#             if isinstance(base_feat, torch.Tensor):
#                 mse_loss = torch.nn.functional.mse_loss(feat, base_feat)
#             else:
#                 mse_loss = torch.nn.functional.mse_loss(feat[0], base_feat[0]) + torch.nn.functional.mse_loss(feat[1], base_feat[1])
#             cls_loss = loss_fn(output, target)
#
#             loss = cls_loss + args.alpha * mse_loss
#
#         if torch.isnan(loss):
#             _write("NaN detected! Stopping training.")
#             return 'NaN'
#
#
#         losses_m_cls.update(cls_loss.item(), input.size(0))
#         losses_m_mse.update(mse_loss.item(), input.size(0))
#
#         optimizer.zero_grad()
#
#         if loss_scaler is not None:
#             assert autocast == amp_autocast
#             loss_scaler(loss, optimizer, parameters=model.parameters())
#         else:
#             assert autocast == suppress
#             loss.backward()
#             optimizer.step()
#
#         lrl = [param_group['lr'] for param_group in optimizer.param_groups]
#         lr = sum(lrl) / len(lrl)
#
#         mem = torch.cuda.max_memory_allocated(device=args.device)
#         mem_mb = torch.tensor([mem / (1024 * 1024 * 1024)], dtype=torch.float, device=args.device)
#         if args.distributed:
#             dist.reduce(mem_mb, 0, op=dist.ReduceOp.SUM)
#
#         if args.local_rank == 0:
#             cur_dur = time.time() - start
#             _write(
#                 '\rTrain Epoch: {:>2d}  Iter: {:>4d}/{}  Eta : {:>4d} min '
#                 'CLS: {cls_loss.val:#.4g} ({cls_loss.avg:#.4g}) MSE: {mse_loss.val:#.4g} ({mse_loss.avg:#.4g})'
#                 'LR: {lr:.3e} GPU mem : local-{mem:.2f} GiB / overall-{mem_overall:.2f} GiB'.format(
#                     epoch,
#                     batch_idx + 1, len(loader), int(cur_dur * ((num_epochs - epoch + 1) * len(loader) - batch_idx-1) / ((((epoch - 1) * len(loader) + batch_idx+1)) * 60.)),
#                     cls_loss=losses_m_cls, mse_loss=losses_m_mse,
#                     lr=lr,
#                     mem=mem / 1024 ** 3, mem_overall=mem_mb.item()), end='')
#     if args.local_rank == 0:
#         _write('')


def validate(model, loader, autocast):
    top1_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):

            input = input.cuda()
            target = target.cuda()

            with autocast('cuda'):
                _, output = model(input)

            acc1, _ = accuracy(output, target, topk=(1, 5))

            top1_m.update(acc1.item(), output.size(0))

        top1_m.synchronize()

    _write('Test  Smples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m))
    return top1_m


if __name__ == '__main__':
    main()
