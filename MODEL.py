
# coding: utf-8

# In[1]:


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets, models
import time
import os
import copy
import torch.optim as optim


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#NEED MATPLOTLIB IN THE ENV.


# ## TRAINING THE MODEL
# 

# ### Create objects to handle and convert image input.

# In[3]:


# #This whole block is likely deprecated
# TRAIN_PATH = "./asl_train"

# #create the transforms.Compose object
# TRANSFORM = transforms.Compose([#20transforms.Resize(64),
#                                 transforms.ToTensor(),
#                                ])
# #Input settings are kind of magic but transform will transform image to 
# #RGB tensor (same as inception, based off original googlenet.)

# #Put all training data into a set using data from root, according to transform
# TRAIN_SET = ImageFolder(root = TRAIN_PATH, transform = TRANSFORM) 

# # Instantiate a dataloader object using that datatset for source.
# # WITH: 10 images/batch, images shuffled to random order, working on 4 threads.
# # Threads no. is kind of a default value
# loader = DataLoader(TRAIN_SET, batch_size = 32, shuffle = True, num_workers = 4)


# In[4]:


# Kind of don't know what this is doing.
# #To test the data inputs:
# for i in range(150):
#     img, label = train_set[-i]
#     print(label)


# In[ ]:


# Define the output classes
classes = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'
          'V', 'W', 'X', 'Y', 'Z')


# In[ ]:


#---------------------MODEL DETAILS-------------------------
# Garcia Viesca: http://cs231n.stanford.edu/reports/2016/pdfs/214_Report.pdf
# ^Starts with a pre-trained GoogLeNet: We're using InceptionV3 it's the closest torch has
# Inception code: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
# This model is really just a feature extraction of Googlenet(inception):
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# It modifies 

#-----------------------------------OLD------------------------------------
# class Model(torch.nn.Module):
#     emptyTensor = torch.empty(299,299) #MEET MINIMUM SIZE 
#     xavier = torch.nn.init.xavier_normal_(emptyTensor)
#     inception =  # create the pretrained I3.
#     # all pre-trained models expect 3channel RGB, batches of shape: (3xHxW)
#     # ------------------H, W MUST BE AT 299!!!!!--------------------------
    
#     def __init__(self):
#         super(Model, self).__init__() #Call SimpleCNN's constructor
        
        
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
#         self.batchnorm1 = nn.BatchNorm2d(8)
#         self.relu = nn.ReLU()                 #RELU Activation
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)   #Maxpooling reduces the size by kernel size. 64/2 = 32
        
#         self.conv2 = 
#         self.pool = torch.nn.MaxPool2d(kernel_size = , stride = , padding =)
    
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#     #batch shape for one input matches googlenet ()
#-----------------------------------OLD--------------------------------------

# Use this to set all the features on the model that you dont want to change
# to non-learning.
def set_requires_grad(model, grad_tf, exclude):
    if grad_tf:
        for param in model.parameters():
            if param.name in exclude:
                param.requires_grad = True
            else: 
                param.requires_grad = False

#used to init weights to xavier tensor.
def init_xavier(tens): #model m, string of type to init, layers in
    #not at all sure if the following method for inits works.
    size = tens.size()
    print(size)
    linear = torch.nn.Linear(size[0], size[1], bias = True)
    torch.nn.init.xavier_normal_(linear.weight)

# The base inceptionv3 model we will be using
model_ft = models.inception_v3(pretrained = True)

# A function for initializing our model, restricted to the layers defined
# in inceptionv3 (one FC layer). FUTURE WORK SHOULD FIND WAY TO IMPLEMENT MORE FC LAYERS.
def initialize_model(model_name, num_classes, feature_extract, reinit_type, use_pretrained = True,):
    # model_name is (inception), num_classes: 26, feature_extract = grad_tf = false, 
    # pretrained = True for Inception, reinit_type = set of weights to init with. ("xavier")
    if model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_requires_grad(model_ft, feature_extract, ["fc"])
        #Following redefines the fc layer to be the layer from Garcia-Viesca
        #The input data for this:
    
        #AUXILARY NET:
        num_in_ftrs = model_ft.AuxLogits.fc.in_features
        print(num_in_ftrs)
        model_ft.AuxLogits.fc = torch.nn.Linear(num_in_ftrs, num_classes)
        torch.nn.init.xavier_normal_(model_ft.AuxLogits.fc.weight)
        
#         model.AuxLogits.fc = torch.nn.Softmax(model.AuxLogits.fc)
        #model_ft.AuxLogits.fc.weight = init_xavier(model_ft.AuxLogits.fc.weight)
        
        #PRIMARY NET:
        num_in_ftrs = model_ft.fc.in_features
        print(num_in_ftrs)
        model_ft.fc = torch.nn.Linear(num_in_ftrs, num_classes)
        torch.nn.init.xavier_normal_(model_ft.fc.weight)
#         model.fc = torch.nn.Softmax(model.fc)
        #model_ft.fc.weight = init_xavier(model_ft.fc.weight)
        
        input_size = 299
        
    return model_ft, input_size
#=========================INITIALIZE THE MODEL=========================

model_ft, input_size = initialize_model("inception", 26, True, "xavier")
CUDA = torch.cuda.is_available()
if CUDA:
    model_ft = model_ft.cuda()
#print(model_ft)
for param in model_ft.named_parameters():
    print(param[1].size())


# ## Train the Network
# 
# ### Constants Box:

# In[ ]:


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./asl_train/*"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"

# Number of classes in the dataset
num_classes = 26

# Batch size for training (change depending on how much memory you have)
batch_size = 20

# Number of epochs to train for
num_epochs = 10

# learning rate:
learning_rate = 0.0000001

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True


# In[ ]:


# Data augmentation and normalization for training
# Just normalization for validation
# The boilerplate doesn't work here, so the dataloader is self made
# Essentially, we just need a train data set and a val dataset in a dictionary,
# named "train" and "val"
# FANCY: Subtract the mean image from the datasets.

empty = torch.zeros(299)
data_transforms = {
    'train': transforms.Compose([
         transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
         transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

TRAIN_PATH = "./asl_train/"
TEST_PATH = "./asl_test/"

#Input settings are kind of magic but transform will transform image to 
#RGB tensor (same as inception, based off original googlenet.)

#Put all training data into a set using data from root, according to transform
TRAIN_SET = ImageFolder(root = TRAIN_PATH, transform = data_transforms["train"])
VAL_SET = ImageFolder(root = TEST_PATH, transform = data_transforms["val"])

# Instantiate a dataloader object using that datatset for source.
# WITH: 10 images/batch, images shuffled to random order, working on 4 threads.
# Threads no. is kind of a default value
dataloaders_dict = {'train': DataLoader(TRAIN_SET, batch_size = batch_size, shuffle = True, num_workers = 4),
                    'val': DataLoader(VAL_SET, batch_size = batch_size, shuffle = True, num_workers = 4)}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ### The Acutal Training Function (Finally, we're doing work!)

# In[ ]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", dataloaders["train"].batch_size)
    print("epochs=", num_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            tracker = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if ((tracker %30 == 0) and (not tracker == 0)):
                    print(running_loss/tracker)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# In[ ]:


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)


# In[ ]:


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
learning_rate = 0.0000001
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

