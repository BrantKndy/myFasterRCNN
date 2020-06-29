import os
from math import floor
from PIL import Image
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Compose

voc_path = \
    r'E:\Projects\Pyprojs\Semantic Segmentation\pytorch-deeplab-xception\dataloaders\datasets\VOC\VOCdevkit\VOC2012'


def transforms(img, target):
    w, h = img.size
    for i, obj_dic in enumerate(target):
        target[i]['xmin'] = floor(obj_dic['xmin'] * 448 / w)
        target[i]['xmax'] = floor(obj_dic['xmax'] * 448 / w)
        target[i]['ymin'] = floor(obj_dic['ymin'] * 448 / h)
        target[i]['ymax'] = floor(obj_dic['ymax'] * 448 / h)

    return ToTensor()(img.resize((448, 448))), target


def collate_fn(batch):
    batch_tag = []
    batch_img = []
    for sample in batch:
        batch_img.append(sample[0])
        batch_tag.append(sample[1])
    return torch.stack(batch_img, 0), batch_tag


class VOC_dataset(Dataset):
    def __init__(self, transforms=None):
        self.root = voc_path
        f = open(os.path.join(self.root, r'ImageSets\Main\train.txt'), 'r')
        self.data_name = f.readlines()
        f.close()
        self.train_jpg_path = os.path.join(self.root, 'JPEGImages')
        self.train_tag_path = os.path.join(self.root, 'Annotations')
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.train_jpg_path, self.data_name[idx].strip()+'.jpg'))
        tree = ET.parse(os.path.join(self.train_tag_path, self.data_name[idx].strip()+'.xml'))
        root = tree.getroot()
        tag = []
        for obj in root.iter('object'):
            obj_dic = {}
            obj_dic['cls'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_dic['xmin'] = int(bbox.find("xmin").text)
            obj_dic['ymin'] = int(bbox.find("ymin").text)
            obj_dic['xmax'] = int(bbox.find("xmax").text)
            obj_dic['ymax'] = int(bbox.find("ymax").text)
            tag.append(obj_dic)

        if self.transforms:
            img, tag = self.transforms(img, tag)
        return img, tag

    def __len__(self):
        return len(self.data_name)


class VOC_Loader:
    def __init__(self, batch_size):
        self.dataset = VOC_dataset(transforms=transforms)
        self.batch_size = batch_size
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=0, collate_fn=collate_fn)




