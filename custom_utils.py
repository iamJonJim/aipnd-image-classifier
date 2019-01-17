#!/usr/bin/env python
# coding: utf-8
# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
# 
# Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.

# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import tensor
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import os
import json
import sys

# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' 

# TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.RandomRotation(45),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
# In[18]:
# TODO: Load the datasets with ImageFolder
#REFERENCE: https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
    image_datasets = [datasets.ImageFolder(train_dir, transform=train_transforms),
                 datasets.ImageFolder(valid_dir, transform=validation_transforms),
                 datasets.ImageFolder(test_dir ,transform=test_transforms)]

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=32, shuffle=True),
              torch.utils.data.DataLoader(image_datasets[1], batch_size=32, shuffle=True),
              torch.utils.data.DataLoader(image_datasets[2], batch_size=32, shuffle=True)]

    return dataloaders, image_datasets

# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

#with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)

# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

# In[20]:
# TODO: Build and train your network
#default_structure = 'vgg'
#dropout = 0.5
#hidden_size = 120
#input_size = 25088
#lr = 0.001
#epochs = 3

def nn_config(dropout, hidden_size, lr, structure = 'vgg', gpu=True):

    # REFERENCE: http://www.robots.ox.ac.uk/~vgg/research/very_deep/        
    
    if structure == 'vgg':
        model = models.vgg19(pretrained = True)
        input_size = 25088
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088

    #for param in model.parameters():
        #param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(dropout)),
                ('fc1', nn.Linear(input_size, hidden_size)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_size, 90)),
                ('relu2', nn.ReLU()),
                ('output', nn.Linear(90, 102)),
                ('softmax', nn.LogSoftmax(dim=1))
                ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if gpu == True:
        model.cuda()
        
    print('End of model configuration')
    
    return model, optimizer, criterion


def nn_train(model, optimizer, criterion, epochs, dataloaders, gpu=True):

    print_every = 10
    steps = 0
    
    if gpu == True:
        model.to('cuda')
        print('Training with GPU')
    else:
        print('Training without GPU')

    for e in range(epochs):
        running_loss = 0
        for xx, (inputs, labels) in enumerate(dataloaders[0]):
            sys.stdout.write('.')
            sys.stdout.flush()
            steps += 1
            
            if gpu == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                lost_valid = 0
                acc_valid = 0

                for xx, (inputs_2, labels_2) in enumerate(dataloaders[1]):
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    optimizer.zero_grad()
                    
                    if gpu == True:
                        inputs_2, labels_2 = inputs_2.to('cuda:0'), labels_2.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs_2)
                        lost_valid = criterion(outputs,labels_2)
                        ps = torch.exp(outputs).data
                        equality = (labels_2.data ==ps.max(1)[1])
                        acc_valid = acc_valid + equality.type_as(torch.FloatTensor()).mean()

                lost_valid = lost_valid / len(dataloaders[1])
                acc_valid = acc_valid / len(dataloaders[1])


                print("\nEpoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost: {:.4f}".format(lost_valid),
                      "Accuracy: {:.4f}".format(acc_valid))

                running_loss = 0
                
# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# TODO: Save the checkpoint 
def save_trained_nn(save_dir, model, optimizer, structure, fc1, lr, dropout, epochs, image_datasets):
    model.class_to_idx = image_datasets[0].class_to_idx
    torch.save({'arch': structure,
                  'learning_rate': lr,
                  'epochs': epochs,
                  'dropout': dropout,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx' : model.class_to_idx},
                   save_dir)

# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    learning_rate = checkpoint['learning_rate']
    structure = checkpoint['arch']
    
    dropout = 0.5
    hidden_size = 120
    model, optimizer, criterion = nn_config(dropout, hidden_size, learning_rate)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']
    
    return model, optimizer, criterion

# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

def process_image(img_filepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model   
    # Load Image
    # REFERENCE: https://pytorch.org/docs/0.3.0/torchvision/transforms.html
    pil_image = Image.open(img_filepath)
    
    width = pil_image.size[0]
    height = pil_image.size[1]
    
    size = 256
    shortest_side = min(width, height)
    
    width = int((width/shortest_side)*size)
    height = int((height/shortest_side)*size)
    pil_image = pil_image.resize((width, height))
                     
    
    # Create data transformer that doesn't include the normalization
    data_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    # Transform
    pil_image = data_transform(pil_image).float()
    np_image = np.array(pil_image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))

    # img_torch = torch.from_numpy(img).float()
    # print(np_image)
    # return img_torch
    return np_image
    
#directory_set_path = data_dir + '/test/1/'
#img_filepath = directory_set_path + os.listdir(directory_set_path)[0]
#np_image = process_image(img_filepath)

# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#imshow(process_image(img_filepath))

# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

def predict(image_filepath, model, topk=5, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    # REFERENCE: https://github.com/miguelangel/ai--transfer-learning-for-image-classification
    np_image = process_image(image_filepath)
    img_torch = Variable(torch.FloatTensor(np_image), requires_grad=True)
    img_torch = img_torch.unsqueeze(0) # this is for VGG
    
    if gpu == True:
        model.to('cuda:0')
    
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torcho.no_grad():
            output = model.forward(img_torch)
        
    probs, idxs = torch.nn.functional.softmax(output.data,dim=1).topk(topk)
#     classes = [cat_to_name[str(index + 1)] for index in np.array(idxs[0])]
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
#     print(index_to_class)
#     print(np.array(idxs))
    top_classes = [index_to_class[each] for each in np.array(idxs[0])]
    probs = np.array(probs)
    print(top_classes)

    return probs, top_classes

#probs, classes = predict(img_filepath, model)
#print(probs)
#print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# TODO: Display an image along with the top 5 classes
#sanity_directory_set_path = data_dir + '/test/1/'
#sanity_img_filename = os.listdir(directory_set_path)[0]
#sanity_img_filepath = directory_set_path + sanity_img_filename

#index = 1
#img = process_image(sanity_img_filepath)
#probs, classes  = predict(sanity_img_filepath, model)

# Image
#axs = imshow(img, ax = plt)
#axs.title(cat_to_name[str(index)])
#axs.axis('off')
#axs.show()

#probalility_values = probs
# probalility_values = np.array(probs[0])
#class_values = [cat_to_name[index] for index in classes]
# class_values = [cat_to_name[str(index + 1)] for index in np.array(classes[0])]
#classes_values_length = float(len(class_values))

#fig , ax = plt.subplots(figsize=(5,3))
#tickLocations = np.arange(classes_values_length)
#ax.barh(tickLocations, probalility_values, align = 'center')

#ax.set_yticks(ticks = tickLocations)
#ax.set_yticklabels(class_values)

# ax.set_xticks(np.linspace(0,1,11))
#ax.set_xlabel('Probabilities')

#plt.show()




