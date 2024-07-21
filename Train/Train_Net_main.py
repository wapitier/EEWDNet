import argparse
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import functional as F
from typing import List, Union
from torch.cuda.amp import autocast, GradScaler

from work_4.Func import transforms as T
from work_4.Func import utils
from work_4.Func import metric_func
from work_4.Func.dataset_D_D import SOD_Dataset
from work_4.Net.Net_main import EEWDNet 


# 随机种子初始化
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

cur_root = '/data2021/tb/AllWork/work_4'
cur_exp_tag = 'Net_main'

all_exp = os.path.join(cur_root, 'Runs')
if not os.path.exists(all_exp):
    os.mkdir(all_exp)

cur_exp_folder = os.path.join(all_exp, cur_exp_tag)
if not os.path.exists(cur_exp_folder):
    os.mkdir(cur_exp_folder)


class SOD_Train:
    def __init__(self, base_size: Union[int, List[int]], hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([ 
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomHorizontalFlip(hflip_prob),
            # T.Resize(base_size, resize_mask=True)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SOD_Eval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # T.Resize(base_size, resize_mask=True)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

    
def bce_iou_loss(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask,size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def eval_duts_seg(args):
    cudnn.benchmark = True

    # building network
    model = EEWDNet(args)
    model = model.cuda()

    # logs file
    results_file = cur_exp_folder + "/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_logs   = cur_exp_folder + "/train_logs{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    # preparing data
    dataset_train = SOD_Dataset(train=True, transform=SOD_Train([args.img_size, args.img_size]))
    sampler = torch.utils.data.RandomSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True
    )

    dataset_val = SOD_Dataset(train=False, transform=SOD_Eval([args.img_size, args.img_size]))
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )

    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'swin' in name or 'resnet' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    params_list = [{'params': backbone_params}, {'params': decoder_params, 'lr': args.lr * 10}]

    # set optimizer
    optimizer = torch.optim.SGD(
        params_list,
        lr=args.lr, 
        momentum=0.9,
        weight_decay=5e-4, 
        nesterov=True
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    to_restore = {"epoch": 0, "best_acc": 0.}
    start_epoch = to_restore["epoch"]
    scaler = GradScaler()

    best_F1 = 0
    for epoch in range(start_epoch, args.epochs):
        lr, loss = train(model, optimizer, train_loader, epoch, scaler)
        scheduler.step()

        with open(train_logs, "a") as f:
                write_info = f"[epoch: {epoch}] " \
                             f"loss: {loss:.3f} lr: {lr:.7f} \n"
                f.write(write_info)

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            mae_info, f1_info = validate_network(model, val_loader, epoch)
            print(f"val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")

            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch}] " \
                             f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} \n"
                f.write(write_info)

            if f1_info >= best_F1:
                best_F1 = f1_info

                save_path = os.path.join(cur_exp_folder, 'best.pth')
                torch.save(model.state_dict(), save_path)
            
    print("Training completed.\n")



def train(model, optimizer, loader, epoch, scaler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, mask in metric_logger.log_every(loader, 20, header):
        # compute the gradients
        optimizer.zero_grad()

        with autocast():
            # move to gpu
            image, mask = image.float().cuda(non_blocking=True), mask.float().cuda(non_blocking=True)
            pred, o1, o2, o3, o4, o5 = model(image)

            loss1 = bce_iou_loss(pred, mask)
            loss2 = bce_iou_loss(o1, mask)
            loss3 = bce_iou_loss(o2, mask)
            loss4 = bce_iou_loss(o3, mask)
            loss5 = bce_iou_loss(o4, mask)
            loss6 = bce_iou_loss(o5, mask)
 
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        
        # log 
        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    ret = []
    for k, meter in metric_logger.meters.items():
        ret.append(meter.global_avg)

    return ret


@torch.no_grad()
def validate_network(model, val_loader, epoch):
    model.eval()

    mae_metric = metric_func.MeanAbsoluteError()
    f1_metric = metric_func.F1Score()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for image, mask in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        image, mask = image.float().cuda(non_blocking=True), mask.float().cuda(non_blocking=True)   

        # forward
        pred, _, _, _, _, _ = model(image)

        mae_metric.update(pred, mask)
        f1_metric.update(pred, mask)


    mae_metric.gather_from_all_processes()
    f1_metric.reduce_from_all_processes()

    mae_info = mae_metric.compute()
    f1_info = f1_metric.compute()
    return mae_info, f1_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=4, type=int, help='Per-GPU batch-size')
    parser.add_argument('--img_size', default=1024, type=int, help='Image size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=1e-4, type=float, help='')
    parser.add_argument('--val_freq', default=5, type=int, help="Epoch frequency for validation.")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1, type=int, help='Number of labels')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--train_mode', default=True)
    
    args = parser.parse_args()
    eval_duts_seg(args)
