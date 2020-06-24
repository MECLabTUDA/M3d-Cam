from torch.utils.data import Dataset
import os
import torch
from tests.test_segmentation.model.utils.dataset import BasicDataset

current_path = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(current_path, 'data/imgs/')
MASK_DIR = os.path.join(current_path, 'data/masks/')

class UnetSegDataset(Dataset):
    def __init__(self, device):
        self.dataset = BasicDataset(IMAGE_DIR, MASK_DIR, 0.3)
        self.device = device

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        item = {"img": item["image"].to(self.device, dtype=torch.float32), "gt": item["mask"].to(self.device, dtype=torch.float32)}
        return item

    def __len__(self):
        return self.dataset.__len__()