from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')

class DIV_2K (Dataset):
    def __init__(self,txt,loader=default_loader,transform=None):
        x = open(txt,'r',encoding='UTF-8')
        imgs = []
        for line in x:
            line = line.strip('\n')
            imgs.append(line)
        # print(imgs)
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        temp = self.imgs[index]
        img = self.loader(temp)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.imgs)
    
# DIV_2K('./training.txt')
