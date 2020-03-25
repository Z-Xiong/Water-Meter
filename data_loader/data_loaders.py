from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np
import torch
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
    def __init__(self, root,trsfm, train=True ):
        self.train = train
        self.root = root
        self.data, self.label = read_images(root, self.train)
        self.transforms = trsfm
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.open(self.root+img)
        # 根据不同图片的大小进行缩放和补零操作，使得水表图像具有相同的高度H和宽度W
        H, W = 48, 160
        ratio = img.height/img.width
        if ratio < H/W:
            h_ = (H-W*ratio)/2
            self.transforms = transforms.Compose([
                transforms.Resize((int(W*ratio), W)),
                transforms.Pad((0, int(h_))),
                transforms.Resize((H, W)),  # 防止每张图片缩放补零后大小不一致
                transforms.ToTensor()
            ])
        else:
            w_ = (W-H*ratio)/2
            self.transforms = transforms.Compose([
                transforms.Resize((H, int(H*ratio))),
                transforms.Pad((int(w_), 0)),
                transforms.Resize((H, W)),
                transforms.ToTensor()
            ])

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
        trsfm = None
        self.data_dir = data_dir
        self.dataset = WaterMeterDataset(self.data_dir, trsfm, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
