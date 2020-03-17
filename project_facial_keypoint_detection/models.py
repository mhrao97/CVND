## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
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
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # second conv layer: 32 inputs, 64 outputs, 3 X 3 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (64, 108, 108)
        # after another pool layer this becomes (64, 54, 54); 
        self.conv2 = nn.Conv2d(32, 64, 3)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # self.pool = nn.MaxPool2d(2, 2)     # using the same as above

        # third conv layer: 64 inputs, 128 outputs, 3 X 3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (128, 52, 52) 
        # after another pool layer this becomes (128, 26, 26); 
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # self.pool = nn.MaxPool2d(2, 2)    # using the same as above

        # forth conv layer: 128 inputs, 256 outputs, 3 X 3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output tensor will have dimensions: (256, 24, 24) 
        # after another pool layer this becomes (256, 12, 12); 
        self.conv4 = nn.Conv2d(128, 256, 3)	
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # self.pool = nn.MaxPool2d(2, 2)     # using the same as above

        # fifth conv layer: 256 inputs, 512 outputs, 1 X 1 conv
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        # the output tensor will have dimensions: (512, 12, 12) 
        # after another pool layer this becomes (512, 6, 6); 
        self.conv5 = nn.Conv2d(256, 512, 1)

        # 512 outputs * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1024)        
        self.fc2 = nn.Linear(1024, 512)
        # finally, create 136 output channels (for the 68 * 2 classes/features)
        self.fc3 = nn.Linear(512, 136)     # ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # dropout with p=0.4
        self.dropout = nn.Dropout(p=0.4)
        
        # batch normalization
        self.bn = nn.BatchNorm2d(128)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #  conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
