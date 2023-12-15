import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.cuda.amp import GradScaler
import logging
from PIL import Image
import torch.nn.functional as F
import numpy as np
import random
def Scheduler(args, optimizer):
    if args.scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.patience)
    elif args.scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)
    return scheduler

def Criterion(args):
    if args.criterion == 'API':
        criterion = api_loss
    elif args.criterion == 'bce':
        criterion = torch.nn.BCELoss()
    return criterion

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr*decay))

def Optimizer(args, model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optimizer

def api_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * (mask)

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    # amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1 + 0.0001).sum(dim=(2, 3))

    return ((abce +aiou + mae)).mean()

class Utils:
    def __init__(self):
        super(Utils, self).__init__()

    @staticmethod
    def collate_fn(batch):
        """Discard none samples.
        """
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    @staticmethod
    def save_images(images: torch.Tensor, save_path: str, nrow: int = 8) -> None:
        grid = torchvision.utils.make_grid(images, nrow=nrow)
        img_arr = grid.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img_arr)
        img.save(save_path)

    @staticmethod
    def save_checkpoint(
            epoch: int,
            model: nn.Module,
            filename: str,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler = None,
            grad_scaler: GradScaler = None,
    ) -> None:
        checkpoint_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }
        if optimizer:
            checkpoint_dict['optimizer'] = optimizer.state_dict()
        if scheduler:
            checkpoint_dict['scheduler'] = scheduler.state_dict()
        if scheduler:
            checkpoint_dict['grad_scaler'] = grad_scaler.state_dict()

        torch.save(checkpoint_dict, filename)
        logging.info("=> Saving checkpoint complete.")

    @staticmethod
    def load_checkpoint(
            model: nn.Module,
            filename: str,
            enable_train_mode: bool,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler = None,
            grad_scaler: GradScaler = None,
    ) -> int:
        logging.info("=> Loading checkpoint")
        saved_model = torch.load(filename, map_location="cuda")
        model.load_state_dict(saved_model['state_dict'], strict=False)
        if 'optimizer' in saved_model and enable_train_mode:
            optimizer.load_state_dict(saved_model['optimizer'])
        if 'scheduler' in saved_model and enable_train_mode:
            scheduler.load_state_dict(saved_model['scheduler'])
        if 'grad_scaler' in saved_model and enable_train_mode:
            grad_scaler.load_state_dict(saved_model['grad_scaler'])
        return saved_model['epoch']

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

class AvgMeter(object):
    def __init__(self, num=10):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))