import os
import argparse
import torch
import json
import copy
import numpy as np
import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse
from datetime import datetime

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

import torch.nn.functional as F

import logging
import random
from torchsummary  import summary
import model as mdl
device = "cpu"
torch.set_num_threads(4)


batch_size = 64 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    totalExecTime =0
    # remember to exit the train loop at end of the epoch
    print(f"Epoch {epoch} triggered")
    for batch_idx, (data, target) in enumerate(train_loader):
         #we are breaking after 40 iterations.
        if(batch_idx ==40):
            break

        #noting the iteration start time
        iterationStartTime= datetime.now().timestamp()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        #calculating the loss
        loss = criterion(outputs, target)
        loss.backward()
        #we are not explicitly doing any function call to gather the gradients from all the nodes DDP handles it internally and updates the weights accordingly.
        optimizer.step()

        #printing loss for every 20 iterations.
        if (batch_idx+1)%20 == 0 :
            print(f"Loss at {batch_idx+1}th batch is {loss.item()}")
            
        #caluclating the execution time.
        execTime =(datetime.now().timestamp()-iterationStartTime) 
        #print(f"Traing time for batch{batch_idx} is : {execTime} minutes")

        if(batch_idx!=0):
            totalExecTime += execTime
        
    print(f"Total execution time is : {totalExecTime} seconds")
    print(f"Average execution time is : {totalExecTime/39} seconds")
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main(rank, nodes):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    
    

    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)

    #distributing the data for all the worker nodes. 
    sampler = torch.utils.data.distributed.DistributedSampler(training_set, \
            rank=rank, num_replicas=nodes, shuffle=False, seed=69143)

    #we are not shuffling the data because we want to test on the same data for all the tasks.  
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    print(summary(model,(3,32,32)))
    #we are directly using distributed data parallel pytorch api with 25MB bucket size.
    model= DDP(model,bucket_cap_mb=25)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)



#Function where the data parallel initialization starts and calls the training and testing method.
def fun_main(master_ip,rank,nodes):
    print(f"master_ip = {master_ip} and nodes = {nodes}")
    #initializating data paralle
    dist.init_process_group("gloo",init_method="tcp://"+master_ip,world_size=nodes, rank=rank, group_name="ass2_workers")
    #setting the seed this will make all the models in the nodes start with the same weights.
    torch.manual_seed(69143)
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    #calling the training and testing method.
    main(rank,nodes)
    #destroying the data parallel initialization
    dist.destroy_process_group()



if __name__ == "__main__":
    #we are getting the params from the args and parsing them.
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip",type=str,default="127.0.1.1:8000")
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--num-nodes",type=int,default=1)
    args = parser.parse_args()
    #calling the function to train and test the data.
    fun_main(args.master_ip, args.rank,args.num_nodes)

