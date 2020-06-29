import os
import torch

import torchvision
import torch.nn as nn
import torch.optim as optim

import argparse

from data import VOC_Loader
from loss import RPNloss
from model.fastrcnn import fastrcnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, criterion):
    for i, (img, target) in enumerate(loader):
        optimizer.zero_grad()
        img = img.to(device)
        # target = target.to(device)
        predict = model(img)
        loss = criterion(predict, target)
        model.backward(loss)
        optimizer.step()


def train(model, loader, optimizer, criterion, epochs):
    for i in range(epochs):
        train_epoch(model, loader, optimizer, criterion)


def main():
    # load paras
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='VOC', help='dataset')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--unique_name', type=str, default='myFasterRCNN',
                        help='unique name for creating unique folder and saving result')
    opt = parser.parse_args()

    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists(os.path.join('output/', opt.unique_name)):
        os.mkdir(os.path.join('output/', opt.unique_name))
    # load data

    loader = VOC_Loader(opt.batchsize).loader
    model = fastrcnn().to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    criterion = RPNloss(0.5)
    train(model, loader, optimizer, criterion, opt.epochs)


if __name__ == '__main__':
    main()