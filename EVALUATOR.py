
# coding: utf-8

# In[1]:


#Imports:
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
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline 
#Model Loader and Inference:


# In[13]:


#The original method definitions

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
        
        #model_ft.AuxLogits.fc = torch.nn.Softmax(model_ft.AuxLogits.fc.weight,1)
        #model_ft.AuxLogits.fc.weight = init_xavier(model_ft.AuxLogits.fc.weight)
        
        #PRIMARY NET:
        num_in_ftrs = model_ft.fc.in_features
        print(num_in_ftrs)
        model_ft.fc = torch.nn.Linear(num_in_ftrs, num_classes)
        torch.nn.init.xavier_normal_(model_ft.fc.weight)
        
        #model_ft.fc = torch.nn.Softmax(model_ft.fc.weight,1)
        #model_ft.fc.weight = init_xavier(model_ft.fc.weight)
        
        input_size = 299
        
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_requires_grad(model_ft, feature_extract,["final_conv"])
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "inception_real":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        #Following redefines the fc layer to be the layer from Garcia-Viesca
        #The input data for this:
    
        #PRIMARY NET:
        num_in_ftrs = model_ft.fc.in_features
        model_ft.AuxLogits.fc = torch.nn.Linear(num_in_ftrs, num_classes)
        torch.nn.init.xavier_normal_(model_ft.AuxLogits.fc.weight)
        input_size = 299
        
    elif model_name == "squeezenet_real":
        model_ft = models.squeezenet1_0(pretrained=True)
        set_requires_grad(model_ft, False,[])
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
        
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    return model_ft, input_size


# In[14]:


model_ft, input_size = initialize_model("squeezenet", 26, True, "xavier")
state_dict = torch.load("./best_epoch_SD.pt")
model_ft.load_state_dict(state_dict)
model_ft.eval()

def infer(model_ft, img):
    size = 224
    TRANSFORM = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = TRANSFORM(img).float()
    img = img.view(-1,3,size,size)
    img.requires_grad = True
    #print(img)
    outputs = model_ft(img)
    macs = torch.argmax(outputs) 
    return chr(macs+97)

