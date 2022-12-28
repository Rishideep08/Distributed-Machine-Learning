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

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from datetime import datetime

import torch.nn.functional as F

import logging
import random
from torchsummary  import summary
import model as mdl
device = "cpu"
torch.set_num_threads(4)

# random_seed = 1262
# torch.manual_seed(random_seed)

batch_size = 64 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    #Variable To calculate the total execution time.
    totalExecTime =0
    # remember to exit the train loop at end of the epoch

    #getting the training data.
    for batch_idx, (data, target) in enumerate(train_loader):
        #we are breaking after 40 iterations.
        if(batch_idx ==40):
            break

        #noting the iteration start time
        iterationStartTime= datetime.now().timestamp()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + gatherAndScatter gradients + optimize
        outputs = model(data)
        #calculating the loss
        loss = criterion(outputs, target)
        loss.backward()
        #using gathering and scattering methods to get the gradients.
        gatherAndScatter(model)

        #updating the weights.
        optimizer.step()

        #printing loss for every 20 iterations.
        if (batch_idx+1)%20 == 0 :
          print(f"Loss at {batch_idx+1}th batch is {loss.item()}")

        #caluclating the execution time.
        execTime =(datetime.now().timestamp()-iterationStartTime)
        #print(f"Traing time for batch{batch_idx+1} is : {execTime} seconds")
        
        #calculating the total execution time.
        if(batch_idx!=0):
            totalExecTime += execTime

        # if(batch_idx == 0):
        #         print(data[0])
   
    print(f"Total execution time is : {totalExecTime} seconds")
    print(f"Average execution time is  : {totalExecTime/39} seconds")
    return None


#function where the gather and scatter happens.
def gatherAndScatter(model):
    scatter_grad_list = None
    for param in model.parameters():
        #getting the grads for each layer.
        grads = torch.as_tensor(param.grad)

        #we are getting the gradients only for node 0.
        if torch.distributed.get_rank() == 0:
            #scatter_grad_list has gradients of a given layer from all the workers in worker O(dst=0)
            scatter_grad_list = [torch.zeros(grads.size()) for i in range(torch.distributed.get_world_size())]

        #Now we are calling the distributed gather function it works for node 0 only since for the remaining the scatter_grad is None.
        torch.distributed.gather(grads, scatter_grad_list,dst=0)

        #We are scattering the data from node 0 to all the nodes.
        if torch.distributed.get_rank() == 0:
            for i in range(1, len(scatter_grad_list)):
                #Update scatter_grad_list[0] :: scatter_grad_list[i] where i>0 has old gradient values
                scatter_grad_list[0] += scatter_grad_list[i]
            #Update scatter_grad_list[i] where i>0 with scatter_grad_list[0]
            scatter_grad_list = [scatter_grad_list[0] for i in range(len(scatter_grad_list))]


        latest_grads = torch.zeros(grads.size())
        # scatter the gradient sum to all the workers.
        torch.distributed.scatter(latest_grads, scatter_grad_list, src=0)
        # update the gradient
        param.grad = latest_grads.detach().clone()

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
            # if(batch_idx == 0):
            #     print(data)

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main(rank,nodes):
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

    #getting the training data.
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)

    #distributing the data for all the worker nodes.                                       
    sampler = torch.utils.data.distributed.DistributedSampler(training_set, \
            rank=rank, num_replicas=nodes, shuffle=False, seed=69143)

    #we are not shuffling the data because we want to test on the same data for all the tasks.
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    #num_workers=4,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              #num_workers=4,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    print(summary(model,(3,32,32)))
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        #training the model
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        #print(list(model.parameters())[0])
        #testing the model.
        test_model(model, test_loader, training_criterion)



#Function where the data parallel initialization starts and calls the training and testing method.
def fun_main(master_ip,rank,nodes):
    print(f"master_ip = {master_ip} and nodes = {nodes}")
    #initializating data parallel.
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
