#load libraries
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
from os import listdir
import json
import argparse

print('Prediction Started','.'*10)

# Set default values for hyperparameters
checkpoint = 'checkpoint.pth'
flocation = 'cat_to_name.json'    
arch=''
ipath = 'flowers/test/100/image_07896.jpg'
topk = 5


# Set up command line arguments
parser = argparse.ArgumentParser()

# Optional arguments for the trained network architecture, chekcpoint and other parameters
parser.add_argument('--topk', dest="topk", action='store', default=5 ,type=int, help='Top X number of classes(Display)')
parser.add_argument('--json', dest="flocation", action='store', default= 'cat_to_name.json',type=str, help='Json file path.')
parser.add_argument('--checkpoint', dest="checkpoint" , action='store', default ='checkpoint.pth',type=str, help='trained model path.')
parser.add_argument('-ipath', dest="ipath", action='store', default='flowers/test/100/image_07896.jpg' ,type=str, help='Image Path')

args = parser.parse_args()

# Select parameters entered in command line
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
with open(flocation, 'r') as f:
    cat_to_name = json.load(f)

def load_trainedmodel(ckp):

    # Load checkpoint
    checkpoint = torch.load(ckp)
    
    # Check model architecture
    model = models.vgg16(pretrained=True)
    in_features = 25088
        
    # Freeze parameters of the pretrained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add updated classifier
    hidden_units = checkpoint['hidden_units']
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden_units)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.15)),
        ('fc2', nn.Linear(hidden_units, 512)),
        ('ReLu2', nn.ReLU()),
        ('Dropout2', nn.Dropout(p=0.15)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # Load saved model state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set model class to index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def tranformation(ipath):
    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the image and apply the transformations
    image = Image.open(ipath)
    image = transform(image)
    
    return image

def prediction(ipath, model, topk):
    
    # Use process_image function to create numpy image tensor
    pyt_image = tranformation(ipath)
    
    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(pyt_image)
    pytorch_tensor = pytorch_tensor.float()
    
    # Removing RunTimeError for missing batch size - add batch size of 1 
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    # Run model in evaluation mode to make predictions
    model.eval()
    LogSoftmax_predictions = model.forward(pytorch_tensor)
    predictions = torch.exp(LogSoftmax_predictions)
    
    # Identify top predictions and top labels
    top_predictions, top_labels = predictions.topk(topk)
    
    
    top_predictions = top_predictions.detach().numpy().tolist()
    
    top_labels = top_labels.tolist()
    
    labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'Flower Type':pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    labels = labels.iloc[top_labels[0]]
    labels['Probabilities'] = top_predictions[0]
    
    return labels

model = load_trainedmodel(checkpoint) 

catogories_topk = prediction(ipath,model,topk)
print(catogories_topk)

print('Prediction Ended','.'*10)