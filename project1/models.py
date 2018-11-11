## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ####### Computationally less expensive model ###########
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.norm4 = nn.BatchNorm2d(256)
#         self.conv5 = nn.Conv2d(256, 128, 1)
#         self.norm5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(30976, 1000)    # 128*5*5 input pixels
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)
        self.drop6 = nn.Dropout(0.6)
        self.drop7 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 136)
        # self.fc4 = nn.Linear(600, 136)
        
        '''
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(36864, 18000)    # 64*24*24 input pixels
        self.fc1_drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(18000, 6000)
        self.fc2_drop = nn.Dropout(0.33)
        self.fc3 = nn.Linear(6000, 600)
        self.fc4 = nn.Linear(600, 136)
        '''
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        ####### Computationally less expensive code #########
        startConv = time.time()
        x = self.drop2(self.pool(self.norm1(F.relu(self.conv1(x)))))
        x = self.drop3(self.pool(self.norm2(F.relu(self.conv2(x)))))
        x = self.drop4(self.pool(self.norm3(F.relu(self.conv3(x)))))
        x = self.drop5(self.pool(self.norm4(F.relu(self.conv4(x)))))
        # x = self.drop6(self.pool(self.norm5(F.relu(self.conv5(x)))))
        endConv = time.time()
        x = x.view(x.size(0), -1)
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        # x = self.drop6(F.relu(self.fc3(x)))
        x = self.fc3(x)
        endFC = time.time()
        print ("Time taken by conv layers: {0}".format(endConv-startConv))
        print ("Time taken by linear layers: {0}".format(endFC - endConv))
        
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        '''
        # a modified x, having gone through all the layers of your model, should be returned
        return x
