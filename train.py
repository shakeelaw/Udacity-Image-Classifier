# load libraries
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from os import listdir
import copy
import argparse


def model_initialization(architecture, hidden_units, learning_rate):
    
    print('Defining the model','.'*5)
    
    # Load the selected architecture
    pretrained_model = getattr(models, architecture)(pretrained=True)
    input_size = pretrained_model.classifier[0].in_features
    
    # Freeze parameters to avoid backpropagation through them
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Build custom classifier
    custom_classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(input_size, hidden_units)),
                           ('relu1', nn.ReLU()),
                           ('dropout1', nn.Dropout(p=0.15)),
                           ('fc2', nn.Linear(hidden_units, 512)),
                           ('relu2', nn.ReLU()),
                           ('dropout2', nn.Dropout(p=0.15)),
                           ('fc3', nn.Linear(512, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))

    # Replace pretrained model's classifier with custom classifier
    pretrained_model.classifier = custom_classifier
    
    # Define loss function, optimizer, and scheduler for training
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(pretrained_model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1, last_epoch=-1)
    
    
    return pretrained_model, loss_function, optimizer, scheduler


def model_training(model, criterion, optimizer, scheduler, num_epochs):

#    since = time.time()
    model.to(device)
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}','.'*5)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
  
            print(f"{phase} Epoch Loss: {epoch_loss:.4f} Ephoch Accuracy: {epoch_acc:.4f}")
            
            # Deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        
        print()
    
    print(f'Best validation Acc: {best_acc:.4f}')

    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model


def check_point_maker(model):
    
    # Save the class to index mapping from the train dataset to the model object
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Move the model to CPU and get its state dictionary
    model.cpu()
    state = model.state_dict()

    # Create the checkpoint dictionary with model architecture, state dictionary,
    # class to index mapping, and other hyperparameters
    checkpoint = {
        'arch': 'vgg16',
        'hidden_units': 5120,
        'state_dict': state,
        'class_to_idx': model.class_to_idx
    }

    # Get the save directory from arguments, or set a default directory
    save_dir = ''
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    # Save the checkpoint dictionary to the specified directory
    torch.save(checkpoint, save_dir)

    

    

# Set default values for hyperparameters
arch = 'vgg16'
hidden_units = 5120
learning_rate = 0.001
epochs = 3
device = 'cpu'

# Set up command line arguments
parser = argparse.ArgumentParser()

# Specify the required data directory
parser.add_argument('data_dir', type=str, help='Directory of Images')

# Optional arguments for the pretrained network architecture, number of hidden units, learning rate, epochs, saving model file, and GPU usage
parser.add_argument('-arch', dest="arch", action="store", default="vgg16", 
                    type=str, help='Select either - vgg16 or densenet121')
parser.add_argument('-hidden_units', dest="hidden_units", action="store", 
                    default=120, type=int, help='Number of Hidden Units')
parser.add_argument('-learning_rate', dest="learning_rate", action="store", 
                    default=0.001, type=float, help='Learning rate of the model')
parser.add_argument('--epochs', dest="epochs", action="store", default=3,
                    type=int, help='Number of Epochs')
parser.add_argument('--save_dir', type=str, help='Saving the model')
#parser.add_argument('-g', '--gpu', dest="device", action="store", default="cpu", 
#                    help='Use Graphical Processing Unit')


# Parse the arguments from command line
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('Directories and transforms definition completed','.'*5)

# Directory location of images
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
######555555
#Define transforms for training and validation sets and normalize images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dictionary holding location of training and validation data
data_dict = {'train':train_dir,
            'valid': valid_dir}

# Images are loaded with ImageFolder and transformations applied
image_datasets = {x: datasets.ImageFolder(data_dict[x],transform = data_transforms[x])
                  for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) 
               for x in ['train', 'valid']}

# Variable used in calculating trining and validation accuracies
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# Variable holding names for classes
class_names = image_datasets['train'].classes

print('Initialization model definition', '.' * 5)

model, criterion, optimizer, scheduler = model_initialization(arch, hidden_units, learning_rate)


print('Model definition Completed','.'*5)

print('Initializing model training','.'*5)

model = model_training(model, criterion, optimizer, scheduler, epochs)


print('Model training completed')

# Call the save_model function with the trained model object as an argument
check_point_maker(model)

print('Your model has been trained and successfully saved.','.'*10)

print(model)

print('End of Simulation','.'*10)
