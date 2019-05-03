from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
plt.ion()   # interactive mode
from collections import OrderedDict

class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor
img_size=224

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'random_output':transforms.Compose([
        transforms.Scale(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ]),
}

data_dir = '/home/sgulshad/sadaf/CUB_experiments/pytorch-nips2017-attack-example'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['random_output']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=False, num_workers=4)
              for x in ['random_output']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['random_output']}
print (dataset_sizes)
class_names = image_datasets['random_output'].classes

use_gpu = torch.cuda.is_available()
# print (class_names)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['random_output']))
print (inputs.shape)
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

model=torch.load("/home/sgulshad/sadaf/CUB_experiments/pytorch-adversarial_box/models/adv_train_PGD_ep16")    
#model=torch.load("transfer_learn_CUB_Stephan")  
model = model.cuda()
#model = torch.nn.DataParallel(model).cuda() 
#Model trained with adversarial training
checkpoint = torch.load("/home/sgulshad/sadaf/CUB_experiments/pytorch-adversarial_box/models/adv_train_PGD_ep16.pth") 
#Model trained on clean images  
#checkpoint = torch.load("transfer_learn_CUB_Stephan.pth")              
new_state_dict = OrderedDict()
#state_dict =checkpoint['state_dict']
state_dict =checkpoint

for i in range(0, len(model.state_dict().keys())):
    new_state_dict[model.state_dict().keys()[i]] = model.state_dict()[model.state_dict().keys()[i]]

model.load_state_dict(new_state_dict)

model.eval()

modules = list( model.children())[-2:]
print(modules)

batch_size=16
total=0
correct=0
counter=0
correct_list=[] ##correctly classified image indexes
 ##incorrectly classified image indexes
incorrect_list=[]
predicted_list=[] ##total list
predicted_incorrect_list=[] ##incorrectly classified classes for each image
predicted_correct_list=[] ##correctly classified classes
images_correct=[]
images_incorrect=[]
predicted_list=[]
for data in dataloaders['random_output']:
    # get the inputs
    images, labels = data
    if use_gpu:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
    else:
        images, labels = Variable(images), Variable(labels)
    outputs = model((images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
#     print (type(predicted1))
#     print (type(labels1))
    predicted1=predicted.cpu()
    predicted1=predicted1.numpy()
    labels1=labels.cpu()
    labels1=labels1.data.numpy()
#     correct += (predicted1 == labels1).sum().item()
# print('Accuracy of the network on the test images: %d %%' % (
#     100 * correct / total))
#     correct += (predicted1 == labels1).sum().item()
    for i in range (len(predicted1)):
        predicted_list.append(predicted1[i])
#         for j in range(len(labels1)):
        if (predicted1[i]==labels1[i]):
#             images_cor   rect.append(images[i])    
            correct_list.append(i+(counter*batch_size))
            predicted_correct_list.append(predicted1[i])
        else:
#             images_incorrect.append(images[i])
            incorrect_list.append(i+(counter*batch_size))
            predicted_incorrect_list.append(predicted1[i])
    counter=counter+1
print('Accuracy of the network on the test images: %d %%' % ( (len(correct_list) / len(predicted_list))*100))


with open ("incorrect_index_random_noise_AT.txt","w") as f:
    np.savetxt(f,zip(incorrect_list,predicted_incorrect_list),fmt='%d')
with open ("correct_index_random_noise_AT.txt","w") as f:
    np.savetxt(f,zip(correct_list,predicted_correct_list),fmt='%d')
print(len(correct_list))
print (len(incorrect_list))
print (len(predicted_list))


