import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.uitls.data as data
import numpy as np
import argparse
from Dataset.COCODataset import get_COCODataLoader
from Model.SSD import  build_SSD
from Dataset.SSD import *

Model_Path = None
data_loader = get_COCODataLoader(batch_size=32,num_workers=2)
model = build_SSD('train',300,101)


if Model_Path!=None:
    model.load_weights(Model_Path)


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
loss = SSD_Loss(101, 0.5, True, 0, True, 3, 0.5,False)

model.train()

loc_loss = 0
conf_loss = 0
epoch = 0

epoch_size = len(dataset) // args.batch_size


for images,targets in data_loader:

    pred = model(images)
    optimizer.zero_grad()
    loss_l,loss_c = loss(pred,targets)
    tot_loss = loss_c+loss_l
    tot_loss.backward()

    optimizer.step()




































