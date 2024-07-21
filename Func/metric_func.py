import torch
import torch.distributed as dist
import torch.nn.functional as F

import errno
import os


def all_gather(data):
    """
    收集各个进程中的数据
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  # 进程数
    if world_size == 1:
        return [data]

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)

    return data_list


class MeanAbsoluteError(object):
    def __init__(self):
        self.mae_list = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        resize_pred = torch.sigmoid(resize_pred)

        error_pixels = torch.sum(torch.abs(resize_pred - gt), dim=(1, 2, 3)) / (h * w)
        self.mae_list.extend(error_pixels.tolist())

    def compute(self):
        mae = sum(self.mae_list) / len(self.mae_list)
        return mae

    def gather_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        gather_mae_list = []
        for i in all_gather(self.mae_list):
            gather_mae_list.extend(i)
        self.mae_list = gather_mae_list

    def __str__(self):
        mae = self.compute()
        return f'MAE: {mae:.3f}'


class F1Score(object):
    """
    refer: https://github.com/xuebinqin/DIS/blob/main/IS-Net/basics.py
    """

    def __init__(self, threshold: float = 0.5):
        self.precision_cum = None
        self.recall_cum = None
        self.num_cum = None
        self.threshold = threshold

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        resize_pred = torch.sigmoid(resize_pred)

        gt_num = torch.sum(torch.gt(gt, self.threshold).float())

        pp = resize_pred[torch.gt(gt, self.threshold)]  # 对应预测map中GT为前景的区域
        nn = resize_pred[torch.le(gt, self.threshold)]  # 对应预测map中GT为背景的区域

        pp_hist = torch.histc(pp, bins=255, min=0.0, max=1.0)
        nn_hist = torch.histc(nn, bins=255, min=0.0, max=1.0)

        # Sort according to the prediction probability from large to small
        pp_hist_flip = torch.flipud(pp_hist)
        nn_hist_flip = torch.flipud(nn_hist)

        pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
        nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

        precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
        recall = pp_hist_flip_cum / (gt_num + 1e-4)

        if self.precision_cum is None:
            self.precision_cum = torch.full_like(precision, fill_value=0.)

        if self.recall_cum is None:
            self.recall_cum = torch.full_like(recall, fill_value=0.)

        if self.num_cum is None:
            self.num_cum = torch.zeros([1], dtype=gt.dtype, device=gt.device)

        self.precision_cum += precision
        self.recall_cum += recall
        self.num_cum += batch_size

    def compute(self):
        pre_mean = self.precision_cum / self.num_cum
        rec_mean = self.recall_cum / self.num_cum
        f1_mean = (1 + 0.3) * pre_mean * rec_mean / (0.3 * pre_mean + rec_mean + 1e-8)
        max_f1 = torch.amax(f1_mean).item()
        return max_f1

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.precision_cum)
        torch.distributed.all_reduce(self.recall_cum)
        torch.distributed.all_reduce(self.num_cum)

    def __str__(self):
        max_f1 = self.compute()
        return f'maxF1: {max_f1:.3f}'


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
