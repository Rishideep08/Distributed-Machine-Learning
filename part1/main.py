import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
from torchsummary  import summary
from datetime import datetime
device = "cpu"
torch.set_num_threads(4)
torch.manual_seed(69143)
batch_size = 256 # batch for one node
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
        #updating the weights.
        optimizer.step()

        #printing the loss for every 20 iterations.
        if (batch_idx+1)%20 == 0 :
            print(f"Loss at {batch_idx+1}th batch is {loss.item()}")
        
        #caluclating the execution time.
        execTime =(datetime.now().timestamp()-iterationStartTime)
        if(batch_idx!=0):
            totalExecTime += execTime
        
    print(f"Total execution time is : {totalExecTime} seconds")
    print(f"Average execution time is  : {totalExecTime/39} seconds")

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
            

def main():
    # I think we need to give data here.
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
    
    #we are getting the data.
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    
    #we are not shuffling the data because we want to test on the same data for all the tasks. 
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    batch_size=batch_size,
                                                    sampler=None,
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
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        #training the model
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        #testing the model
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
