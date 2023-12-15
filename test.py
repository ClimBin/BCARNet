import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import  ICAN
import cv2
from dataloader import TestDatasetFactory
from torch.utils.data import DataLoader
import imageio
import matplotlib.pyplot as plt


class Test(object):
    def __init__(self, cfg):
        #self.heatpath = cfg.heatpath
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = cfg.save_path 

        self.model = ICAN(cfg).to(self.device)
        
        path = "checkpoint.pth"  
        self.model.load_state_dict(torch.load(path),strict=False)

        test_data = TestDatasetFactory(args=cfg)
        self.test_dataloader = DataLoader(test_data, 
                                  batch_size=cfg.batch_size,
                                  drop_last=False)
        
        self.test_processing()

    def test_processing(self):
        self.model.eval()
        with torch.no_grad():
            for i, (images, masks, image_name, size) in enumerate(self.test_dataloader):
                images = images.cuda()
                a,_,_,_= self.model(images)

                H, W = size
                for i in range(images.size(0)):
                    h, w = H[i].item(), W[i].item()
                    
                    output = F.upsample(a[i].unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)
                    output = output.sigmoid().data.cpu().numpy().squeeze()
                    output = (output - output.min()) / (output.max() - output.min() + 1e-8) * 255
                    path = os.path.join('/data/preds/our_4199', image_name[i])


class Test_all(object):
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = cfg.save_path 

        self.model = ICAN(cfg).to(self.device)

        base_path = r"/data/ors4_ckpts/"

        test_data = TestDatasetFactory(args=cfg)
        self.test_dataloader = DataLoader(test_data, 
                                  batch_size=cfg.batch_size,
                                  drop_last=False)

        ckpt_list = os.listdir(base_path)
        for ckpt in ckpt_list:
            if ckpt.endswith('pth') == True:
                print(ckpt)
                model_path = os.path.join(base_path,ckpt)
                self.model.load_state_dict(torch.load(model_path))
                pass_path = os.path.join(cfg.save_path,ckpt[0:-4])
                if not os.path.exists(pass_path):
                    os.makedirs(pass_path)
                self.test_processing(pass_path)

    def test_processing(self,path):
        self.model.eval()
        with torch.no_grad():
            for i, (images, masks, image_name, size) in enumerate(self.test_dataloader):
                images = images.cuda()
                a,_,_,_= self.model(images)

                H, W = size
                for i in range(images.size(0)):
                    h, w = H[i].item(), W[i].item()
                    output = F.upsample(a[i].unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)
                    output = output.sigmoid().data.cpu().numpy().squeeze()
                    output = (output - output.min()) / (output.max() - output.min() + 1e-8) * 255
                    path_l = os.path.join(path, image_name[i])
                    cv2.imwrite(path_l, output)

        
        

import argparse
parser = argparse.ArgumentParser()

#dataset setting
parser.add_argument("--root_dir", type=str, default=r'/data/Datasets/ORS-4199/')
parser.add_argument("--val_root_dir", type=str, default=r'/data/Datasets/ORS-4199/')
parser.add_argument("--masks_dir", type=str, default='masks')
parser.add_argument("--images_dir", type=str, default='images')
parser.add_argument("--image_size", type=int, default=384)

#model setting
parser.add_argument("--backbone_name", type=str, default='pvt')
parser.add_argument("--channels", type=list, default=[64,128,320,512])

#train setting
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--mode", type=str, default='test')
parser.add_argument("--use_clip", type=bool, default=True)
parser.add_argument("--save_path", type=str, default=r'/...')
parser.add_argument("--train_pred_path", type=str, default=r'/...')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mixed_precision", type=str, default="fp16")  # no or yes
parser.add_argument("--num_workers", type=int, default=4)


args = parser.parse_args()
Test(args)
#Test_all(args)