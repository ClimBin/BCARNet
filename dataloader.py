import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from typing import Tuple
import cv2

class TrainDatasetFactory(Dataset):
    def __init__(self, args):
        super(TrainDatasetFactory, self).__init__()

        self.root_dir = args.root_dir
        self.masks_dir = args.masks_dir
        self.images_dir = args.images_dir
        self.image_size = args.image_size

        self.root_dir = os.path.join(self.root_dir,'trainset')

        self.masks_list = sorted(os.listdir(os.path.join(self.root_dir,self.masks_dir)))
        self.images_list = sorted(os.listdir(os.path.join(self.root_dir,self.images_dir)))

        self.transform_imges = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225],
            ),
        ])
        self.transform_masks = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
        ])

        self.image_size = len(self.masks_list)

    def __len__(self) -> int:
        return self.image_size

    def __getitem__(self, idx: int):
        masks_path = os.path.abspath(os.path.join(self.root_dir,self.masks_dir,self.masks_list[idx]))
        images_path = os.path.abspath(os.path.join(self.root_dir,self.images_dir,self.images_list[idx]))
        masks = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)
        # masks = masks[None,...]
        images = cv2.imread(images_path, cv2.COLOR_BGR2RGB)
        return  self.transform_imges(images), self.transform_masks(masks)
    

class TestDatasetFactory(Dataset):
    def __init__(self, args):
        super(TestDatasetFactory, self).__init__()

        self.mode = args.mode  
        self.root_dir = args.val_root_dir
        self.masks_dir = args.masks_dir
        self.images_dir = args.images_dir
        self.image_size = args.image_size

        self.root_dir = os.path.join(self.root_dir,'testset')

        self.masks_list = sorted(os.listdir(os.path.join(self.root_dir,self.masks_dir)))
        self.images_list = sorted(os.listdir(os.path.join(self.root_dir,self.images_dir)))

        self.transform_imges = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225],
            ),
        ])
        self.transform_masks = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
        ])

        self.image_size = len(self.masks_list)

    def __len__(self) -> int:
        return self.image_size

    def __getitem__(self, idx: int):
        masks_path = os.path.abspath(os.path.join(self.root_dir,self.masks_dir,self.masks_list[idx]))
        images_path = os.path.abspath(os.path.join(self.root_dir,self.images_dir,self.images_list[idx]))
        mask_name = masks_path.split('/')[-1]
        masks = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)
        size = masks.shape
        # masks = masks[None,...]
        images = cv2.imread(images_path, cv2.COLOR_BGR2RGB)
        return  self.transform_imges(images), self.transform_masks(masks), mask_name, size
    

