import torch
import numpy as np
from math import sqrt, floor
import torchvision.models as models
import torch.nn as nn


class fastrcnn(nn.Module):
    def __init__(self):
        super(fastrcnn, self).__init__()
        self.conv1 = models.resnet18(pretrained=True)
        self.conv1 = nn.Sequential(*list(self.conv1.children())[:-2])
        self.RPN = RPN()
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
        )
        self.fc_reg = nn.Conv2d(512, 4, 1)
        self.fc_cls = nn.Sequential(
            nn.Conv2d(512, 20, 1),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        ROIs = self.RPN(x)
        ROIspooled = self.ROIPooling(x, ROIs)
        for rpp in ROIspooled:
            bbox = self.fc_reg(rpp)
            cls = self.fc_cls(rpp)
        return bbox, cls

    def ROIPooling(self, x, bboxes):
        return x


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        self.pixelwise1 = nn.Conv2d(512, 2*9, 1)
        self.pixelwise2 = nn.Conv2d(512, 4*9, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        anchors = self.genanchors(x)
        x_gap = self.pixelwise2(x)
        x = self.pixelwise1(x)
        x = x.reshape((-1, 2, 9, 14, 14))
        x = self.softmax(x)
        x = x.reshape((-1, 18, 14, 14))
        x = x+x_gap

    def genanchors(self, x):
        scale = [3, 5, 7]
        ratio = [(1, 2), (floor(sqrt(2)), floor(sqrt(2))), (2, 1)]
        combination = [(i*32*j[0], i*32*j[1]) for i in scale for j in ratio]
        anchor_num = 0
        for anchor_size in combination:
            anchor_num += len(range(int(-anchor_size[0]/2), int(448-anchor_size[0]/2), 32)) * \
                          len(range(int(-anchor_size[1]/2), int(448-anchor_size[1]/2), 32))
        anchors = np.zeros((anchor_num, 4))
        idx_start = 0
        for anchor_size in combination:
            idx_end = idx_start+len(range(int(-anchor_size[0]/2), int(448-anchor_size[0]/2), 32)) * \
                          len(range(int(-anchor_size[1]/2), int(448-anchor_size[1]/2), 32))
            anchors[idx_start:idx_end] = np.array([np.array([xmin, ymin, xmin+anchor_size[0], ymin+anchor_size[1]])
                      for xmin in range(int(-anchor_size[0]/2), int(448-anchor_size[0]/2), 32)
                      for ymin in range(int(-anchor_size[1]/2), int(448-anchor_size[1]/2), 32)])
            idx_start = idx_end
        anchors[np.where(anchors < 0)] = 0
        anchors[np.where(anchors > 447)] = 447

        return anchors

