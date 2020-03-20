from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np
import torch
import os
from PIL import Image

def read_images(root, train=True):
    txt_fname = root+('difficult_samples_for_')+('train.txt' if train else 'test.txt')
    images = []
    labels = []
    with open(txt_fname, 'r') as f:
        for line in f.readlines():
            images.append(line.split()[0])
            labels.append(line.split()[1])
    data = [i for i in images]
    label = [i for i in labels]

    return data, label

class WaterMeterDataset(Dataset):
    def __init__(self, root,transforms, train=True ):
        self.train = train
        self.root = root
        self.data, self.label = read_images(root, self.train)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.open(self.root+img).convert('RGB')
        img = self.transforms(img)
        label = self.label[index]  #获得字符串'0,1,2,5,14'
        label = label.split(',')  #获得列表['0','1','2','5','14']
        label = [int(i) for i in label]  # 转换为[0, 1, 2, 5, 4]
        label = torch.from_numpy(np.array(label)).long()
        
        return img, label

    def __len__(self):
        return len(self.data)


class WaterMeterDataLoader(BaseDataLoader):
    """
    WaterMeter loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((48, 160)),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = WaterMeterDataset(self.data_dir, trsfm, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)