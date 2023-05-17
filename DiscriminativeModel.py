import numpy as np, pandas as pd, os, torch
from pathlib import Path
import cv2
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from statistics import mode
from collections import OrderedDict
import json

from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, downsample):

        super(ResBlock, self).__init__()

        if downsample:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = torch.nn.Sequential()

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):

        shortcut = self.shortcut(input)
        input = torch.nn.functional.relu(self.bn1(self.conv1(input)))
        input = torch.nn.functional.relu(self.bn2(self.conv2(input)))
        input = input + shortcut
        return torch.nn.functional.relu(input)

class ResNet(torch.nn.Module):

  def __init__(self, in_channels=3, outputs=10):

    super(ResNet, self).__init__()

    self.layer0 = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(inplace=True),

    torch.nn.Conv2d(64, 128, kernel_size=3,  padding=1),
    torch.nn.BatchNorm2d(128),
    torch.nn.ReLU(inplace=True),
    torch.nn.MaxPool2d(2)
    )

    self.layer1 = ResBlock(128, 128, downsample=False)


    self.layer2 = torch.nn.Sequential(

        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2),

        torch.nn.Conv2d(256, 512, kernel_size=3,  padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2)
    )

    self.layer3 = torch.nn.Sequential(ResBlock(512, 512, downsample=False),
        torch.nn.MaxPool2d(4),
        torch.nn.Flatten()
    )
    self.fc = torch.nn.Sequential(torch.nn.Linear(512, outputs))

    self.loss_criterion = torch.nn.CrossEntropyLoss() 

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, x, get_encoding=False):

    y_hat = self.layer0(x)
    y_hat = self.layer1(y_hat)
    y_hat = self.layer2(y_hat)
    y_hat = self.layer3(y_hat)
    if get_encoding:
      return y_hat
    y_hat = self.fc(y_hat)

    return y_hat
  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr, epoches, steps_per_epoch, decay=1e-4):

    # opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience=5)

    opt = torch.optim.Adam(self.parameters(), lr, weight_decay=1e-4)
    print(steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epoches, 
                                              steps_per_epoch=steps_per_epoch)
  
    return opt, scheduler

  def computeLoss(self, outputs, data):
    return self.loss_criterion(outputs, data['labels'].to(self.device) )

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

def train_model( trainloader, devloader, epoches, batch_size, lr, output):

  SS = []
  eerror, ef1, edev_error, edev_f1, eloss, dev_loss= [], [], [], [], [], []
  best_f1 = None

  model = ResNet()
  optimizer, scheduler = model.makeOptimizer(lr=lr, epoches=epoches, steps_per_epoch=len(trainloader))

  for epoch in range(epoches):
    running_stats = {'preds': [], 'labels': [], 'loss': 0.}

    model.train()

    iter = tqdm(enumerate(trainloader, 0))
    iter.set_description(f'Epoch: {epoch:3d}')
    for j, data_batch in iter:

      torch.cuda.empty_cache()         
      inputs, labels = data_batch    
      
      
      outputs = model(inputs.to('cuda'))
      loss = model.loss_criterion(outputs, labels.to('cuda'))
   
      loss.backward()

      torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)

      optimizer.step()
      optimizer.zero_grad()

      scheduler.step()
      SS += [get_lr(optimizer)]
      # print statistics
      with torch.no_grad():
        
        running_stats['preds'] += torch.max(outputs, 1)[1].detach().cpu().numpy().tolist()
        running_stats['labels'] += labels.detach().cpu().numpy().tolist()
        running_stats['loss'] += loss.item()
        
        f1 = f1_score(running_stats['labels'], running_stats['preds'], average='macro')
        error = 1. - accuracy_score(running_stats['labels'], running_stats['preds'])
        loss = running_stats['loss'] / (j+1)

      iter.set_postfix_str(f'loss:{loss:.3f} f1:{f1:.3f}, error:{error:.3f}') 

      if j == len(trainloader) - 1:
      
        model.eval()
        eerror += [error]
        ef1 += [f1]
        eloss += [loss]

        with torch.no_grad():
          
          running_stats = {'preds': [], 'labels': [], 'loss': 0.}
          for k, data_batch_dev in enumerate(devloader, 0):
            torch.cuda.empty_cache() 

            inputs, labels = data_batch_dev    
            outputs = model(inputs.to('cuda'))

            running_stats['preds'] += torch.max(outputs, 1)[1].detach().cpu().numpy().tolist()
            running_stats['labels'] += labels.detach().cpu().numpy().tolist()

            loss = model.loss_criterion(outputs, labels.to('cuda'))
            running_stats['loss'] += loss.item()
          

          f1 = f1_score(running_stats['labels'], running_stats['preds'], average='macro')
          error = 1. - accuracy_score(running_stats['labels'], running_stats['preds'])
          loss  = running_stats['loss'] / len(devloader)
          
          edev_error += [error]
          edev_f1 += [f1]
          dev_loss += [loss]

        if best_f1 is None or best_f1 < edev_error:
          torch.save(model.state_dict(), output) 
          best_f1 = edev_error
        iter.set_postfix_str(f'loss:{eloss[-1]:.3f} f1:{ef1[-1]:.3f} error:{eerror[-1]:.3f} dev_loss: {loss:.3f} f1_dev:{f1:.3f} dev_error:{error:.3f}') 
        
  return {'error': eerror, 'f1': ef1, 'dev_error': edev_error, 'dev_f1': edev_f1}, SS

    

def load_dataset(batch_size):

  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ]
      )

  transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)
      ])

  train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
                                            batch_size=batch_size,
                                            shuffle=True)

  dev_loader = DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
                                          batch_size=batch_size,
                                          shuffle=False)
  
  return train_loader, dev_loader
