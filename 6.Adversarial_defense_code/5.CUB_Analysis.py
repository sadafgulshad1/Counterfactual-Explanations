from __future__ import division
import os
import pandas as pd
import imageio
print (imageio.__version__)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset
from collections import OrderedDict
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#cpu_device = torch.device('cpu')
from torch.autograd import Variable
import sys
from sklearn.preprocessing import normalize
import numpy as np
import h5py
import math
from random import shuffle
import random
import csv
from scipy.misc import toimage
from scipy.spatial import distance
from sklearn import metrics
from sklearn.metrics import pairwise
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from textwrap import wrap

#random.seed(5)
use_gpu = torch.cuda.is_available()

data_path = ''
name = 'cub_resnet152'
train = True
# eval model is only used when train=False
eval_model = '{}.pth'.format(name)

# Hyper Parameters
num_epochs = 100
batch_size = 16
learning_rate = 0.001
num_workers = 4
num_classes = 200
step_size = 20
weight_decay = 0
out_freq = 10
# DATASET
#Loading attributes and saving them in an array (ATTRIBUTES_temp)
ATTRIBUTES_temp =[]
with open('/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt') as inputfile:
    for row in csv.reader(inputfile):
        ATTRIBUTES_temp.append(row[0])
inputfile.closed
ATTRIBUTES_temp=np.array(ATTRIBUTES_temp)
print (ATTRIBUTES_temp.shape)
#Each class have 312 dimensional attribute vector
ATTRIBUTES=np.zeros((200,312))
for i in range(200):
    str1=ATTRIBUTES_temp[i].split(' ')
    #print(str1)
    for j, content in enumerate(str1):
        #print (j)
        #print(content)
        ATTRIBUTES[i,j]= content
print (ATTRIBUTES.shape)
# print (ATTRIBUTES[0])
from sklearn import preprocessing
ATTRIBUTES=preprocessing.normalize(ATTRIBUTES)


class CUB(Dataset):
    """CUB200-2011 dataset."""
    def __init__(self, root, train=True, transform=None, normalize=True,download=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data_dir = ( '/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/CUB_200_2011/images')
        train_test_split = pd.read_csv(os.path.join(self.root, '/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/CUB_200_2011/train_test_split.txt'),sep=' ', index_col=0, header=None)
        if train:
            is_train_image = 1
        else:
            is_train_image = 0
        self.img_ids = train_test_split[train_test_split[1] == is_train_image].index.tolist()
        self.id_to_img = pd.read_csv(( '/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/CUB_200_2011/images.txt'),
                sep=' ', index_col=0, header=None)

    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_name = self.id_to_img[self.id_to_img.index == img_id].values[0][0]
        img_path = os.path.join(self.data_dir, img_name)
        img = imageio.imread(img_path, pilmode='RGB')
        label = int(img_name[:3]) - 1
        if self.transform:
            img = self.transform(img)
        return img, label
##class for /transformation/loading of adv images
class CUB_adv(Dataset):
    """CUB200-2011 dataset."""
    def __init__(self, root, train=True, transform=None, normalize=True,download=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data_dir = ( '/home/sgulshad/sadaf/CUB_experiments/pytorch-nips2017-attack-example/output_adv_test/')
        train_test_split = pd.read_csv(os.path.join(self.root, '/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/CUB_200_2011/train_test_split.txt'),sep=' ', index_col=0, header=None)
        if train:
            is_train_image = 1
        else:
            is_train_image = 0
        self.img_ids = train_test_split[train_test_split[1] == is_train_image].index.tolist()
        self.id_to_img = pd.read_csv(( '/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/CUB_200_2011/images.txt'),
                sep=' ', index_col=0, header=None)

    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_name = self.id_to_img[self.id_to_img.index == img_id].values[0][0]
        img_path = os.path.join(self.data_dir, img_name)
        img = imageio.imread(img_path, pilmode='RGB')
        label = int(img_name[:3]) - 1
        if self.transform:
            img = self.transform(img)
        return img, label
transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomSizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                           std=(0.229, 0.224, 0.225))])
transform_test = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Scale(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                           std=(0.229, 0.224, 0.225))])
transform_test_adv = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Scale(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                           std=(0.229, 0.224, 0.225))])

train_dataset = CUB(root=data_path,
                    train=True,
                    transform=transform_train,
                    download=True)

test_dataset = CUB(root=data_path,
                   train=False,
                   transform=transform_test)

test_dataset_adv = CUB_adv(root=data_path,
                   train=False,
                   transform=transform_test_adv)


val_size = int(len(train_dataset) * 0.1)
train_size = len(train_dataset) - val_size
print ("training set size",train_size)
print ("val set size",val_size)
print ("test set size",len(test_dataset))
train_dataset, val_dataset = torch.utils.data.dataset.random_split(train_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers)
test_loader_adv = torch.utils.data.DataLoader(dataset=test_dataset_adv,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers)

att_pred_t=np.loadtxt('out_attr_test.txt')
att_pred_t=att_pred_t.reshape(-1,312)

att_pred_adv=np.loadtxt('out_attr_test_adv.txt')
# print (att_pred_t.shape)
att_pred_adv=att_pred_adv.reshape(-1,312)

# %store -r imgs_test_adv
# print (imgs_test_adv.dtype)
# print (test_images_adv[0])
clean_img_label=np.loadtxt('out_test.txt')
print(clean_img_label.shape)
adv_img_label=np.loadtxt('out_test_adv.txt')
print (adv_img_label.shape)

att_names=[]
att_index=[]
with open('/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/attributes.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names.append(row[0])
        att_index.append(row[0].split(' ')[0])
inputfile.closed
att_names=np.array(att_names)
att_index=[int(x) for x in att_index]
att_index= (np.asarray(att_index))

## classes
class_name =[]
class_index=[]
with open ("/home/sgulshad/sadaf/CUB_experiments/CUB_200_2011/CUB_200_2011/classes.txt") as inputfile:
    for row in csv.reader(inputfile):
        class_name.append(row[0].split(' ')[1])
        class_index.append(row[0].split(' ')[0])
inputfile.closed

SJE_correct_index=[]
SJE_correct_label=[]
SJE_incorrect_index=[]
SJE_incorrect_label=[]
with open('SJE_correct_index.txt') as inputfile:
    for row in csv.reader(inputfile):
#         print (row)
        SJE_correct_index.append(row[0].split(' ')[0])
        SJE_correct_label.append(row[0].split(' ')[1])
print (len(SJE_correct_label))
with open('SJE_incorrect_index.txt') as inputfile:
    for row in csv.reader(inputfile):
        SJE_incorrect_index.append(row[0].split(' ')[0])
        SJE_incorrect_label.append(row[0].split(' ')[1])
print (len(SJE_incorrect_label))
for i in range (len(SJE_correct_label)):
    SJE_correct_label[i]=SJE_correct_label[i].rstrip()

for i in range (len(SJE_incorrect_label)):
    SJE_incorrect_label[i]=SJE_incorrect_label[i].rstrip()

for i in range (len(SJE_correct_index)):
    SJE_correct_index[i]=SJE_correct_index[i].lstrip()

for i in range (len(SJE_incorrect_index)):
    SJE_incorrect_index[i]=SJE_incorrect_index[i].lstrip()

# print (SJE_correct_label)
SJE_correct_index=[float(x) for x in SJE_correct_index ]
SJE_correct_label=[float(x) for x in SJE_correct_label ]
SJE_incorrect_index=[float(x) for x in SJE_incorrect_index ]
SJE_incorrect_label=[float(x) for x in SJE_incorrect_label ]
SJE_correct_index=[int(x) for x in SJE_correct_index]
SJE_correct_label=[int(x) for x in SJE_correct_label ]
SJE_incorrect_index=[int(x) for x in SJE_incorrect_index ]
SJE_incorrect_label=[int(x) for x in SJE_incorrect_label ]

SJE_correct_index_adv=[]
SJE_correct_label_adv=[]
SJE_incorrect_index_adv=[]
SJE_incorrect_label_adv=[]
with open('SJE_correct_index_adv.txt') as inputfile:
    for row in csv.reader(inputfile):
#         print (row)
        SJE_correct_index_adv.append(row[0].split(' ')[0])
        SJE_correct_label_adv.append(row[0].split(' ')[1])
with open('SJE_incorrect_index_adv.txt') as inputfile:
    for row in csv.reader(inputfile):
        SJE_incorrect_index_adv.append(row[0].split(' ')[0])
        SJE_incorrect_label_adv.append(row[0].split(' ')[1])

for i in range (len(SJE_correct_label_adv)):
    SJE_correct_label_adv[i]=SJE_correct_label_adv[i].rstrip()

for i in range (len(SJE_incorrect_label_adv)):
    SJE_incorrect_label_adv[i]=SJE_incorrect_label_adv[i].rstrip()

for i in range (len(SJE_correct_index_adv)):
    SJE_correct_index_adv[i]=SJE_correct_index_adv[i].lstrip()

for i in range (len(SJE_incorrect_index_adv)):
    SJE_incorrect_index_adv[i]=SJE_incorrect_index_adv[i].lstrip()

# print (SJE_correct_label_adv)
SJE_correct_index_adv=[float(x) for x in SJE_correct_index_adv ]
SJE_correct_label_adv=[float(x) for x in SJE_correct_label_adv ]
SJE_incorrect_index_adv=[float(x) for x in SJE_incorrect_index_adv ]
SJE_incorrect_label_adv=[float(x) for x in SJE_incorrect_label_adv ]
SJE_correct_index_adv=[int(x) for x in SJE_correct_index_adv ]
SJE_correct_label_adv=[int(x) for x in SJE_correct_label_adv ]
SJE_incorrect_index_adv=[int(x) for x in SJE_incorrect_index_adv ]
SJE_incorrect_label_adv=[int(x) for x in SJE_incorrect_label_adv ]


##SELECTING THE IMAGES WHICH WHEN CLEAN ARE CLASSIFIED CORRECTLY AND WHEN PERTURBED GET MISSCLASSIFIED ##
##If image index in correctly classified list and not in correctly classified adversarial list this means this image is missclassified
not_same = [item for item in SJE_correct_index if item in SJE_incorrect_index_adv]
##Indexes of all those clean images which are missclassified when perturbed
index_not_same=[]
for i in (not_same):
    index_not_same.append(SJE_correct_index.index(i))
print ("###### Correct labels ######")
### Correct labels: correct labels of clean images, correct_image_no: image no's of clean images, correct_index: indexes of clean images ###
correct_labels=[]
correct_image_no=[]
incorrect_labels=[]
incorrect_image_no=[]
for j in index_not_same:
    correct_labels.append(int(SJE_correct_label[j]))
    correct_image_no.append(int(SJE_correct_index[j]))
correct_index=index_not_same
print ("total no of correct labels",len(correct_labels))
print (correct_image_no[0:10])
print ("###### Incorrect labels ######")
index_same=[]
same=[item for item in not_same if item in SJE_incorrect_index_adv]
for i in  ((not_same)):
    index_same.append(SJE_incorrect_index_adv.index(i))
### incorrect labels: incorrect labels of adv images, incorrect_image_no: image no's of adv images, incorrect_index: indexes of adv images ###
for j in index_same:
    incorrect_labels.append(int(SJE_incorrect_label_adv[j]))
    incorrect_image_no.append(int(SJE_incorrect_index_adv[j]))
print ("total no of incorrect labels",len(incorrect_labels))
incorrect_index=index_same
print (incorrect_image_no[0:10])

#### SELECTING THE IMAGES WHICH ARE CLASSIFIED INCORRECTLY FOR WITHOUT ADVERSARIAL TRAINING AND EVEN WITH ADVERSARIAL TRAINING ####
not_same_AT=[item for item in SJE_incorrect_index if item in SJE_incorrect_index_adv]
index_not_same_AT=[]
index_not_same_NoAT=[]
for i in (not_same_AT):
    index_not_same_AT.append(SJE_incorrect_index.index(i))
    index_not_same_NoAT.append(SJE_incorrect_index_adv.index(i))
incorrect_labels_AT=[]
incorrect_image_no_AT=[]
ATT_PRED_ADV_I=[]
ATT_PRED_ADV_AT_I=[]
for j in index_not_same_AT:
    incorrect_labels_AT.append(int(SJE_incorrect_label[j]))
    incorrect_image_no_AT.append(int(SJE_incorrect_index[j]))
incorrect_index_AT=index_not_same_AT
incorrect_index_NoAT=index_not_same_NoAT
print ("total no of incorrect labels",len(incorrect_labels_AT))

for j in incorrect_index_NoAT:
    ATT_PRED_ADV_I.append(att_pred_adv[j])
for j in incorrect_index_AT:
   ATT_PRED_ADV_AT_I.append(att_pred_t[j])

### Most discriminative Attributes selection using ###

att_index_pred1=[]
scale=25
name_save=[]
class_save=[]
class_adv_save=[]
att_pred_t_save=[]
att_pred_adv_save=[]
name_save_high=[]
name_save_high_adv=[]
att_pred_high_save=[]
att_pred_high_adv_save=[]
ATT_PRED_ADV=[]
ATT_PRED_T=[]
for j in incorrect_index:
    class_adv_save.append(class_name[int(SJE_incorrect_label_adv[j])])
    ATT_PRED_ADV.append(att_pred_adv[j])
for j in correct_index:
    class_save.append(class_name[int(SJE_correct_label[j])])
    ATT_PRED_T.append(att_pred_t[j])

ATT_PRED_T=np.asarray(ATT_PRED_T)
ATT_PRED_ADV=np.asarray(ATT_PRED_ADV)
for i in range (len(ATT_PRED_T)):
    name_save.append(att_names[np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:]])

    pred_att_clean_value=ATT_PRED_T[i,np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:]]
    pred_att_adv_value=ATT_PRED_ADV[i,np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:]]
    diff_temp=ATT_PRED_T[i]-ATT_PRED_ADV[i]
    diff_pred = (np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:])
    diff_pred_value = diff_temp[(np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:])]


    att_index_pred1.append(att_index[np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:]])
    att_pred_t_save.append(ATT_PRED_T[i,att_index[(np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:])]-1])
    att_pred_adv_save.append(ATT_PRED_ADV[i, att_index[(np.argpartition((ATT_PRED_T[i]-ATT_PRED_ADV[i]),-scale)[-scale:])]-1])

    #print ("########### Highest clean attr values ###########")
    name_save_high.append(att_names[np.argpartition((ATT_PRED_T[i]),-scale)[-scale:]])
    att_pred_high_save.append (ATT_PRED_T[i,np.argpartition((ATT_PRED_T[i]),-scale)[-scale:]])
    #print ("########### Highest adv attr values ###########")
    name_save_high_adv.append(att_names[np.argpartition((ATT_PRED_ADV[i]),-scale)[-scale:]])
    att_pred_high_adv_save.append (ATT_PRED_ADV[i, np.argpartition((ATT_PRED_ADV[i]),-scale)[-scale:]])

##Saving top most changing attributes##
with open('discriminative_attr_all.txt','wb') as f:
    for i in  (range(len(name_save))):

        np.savetxt (f,zip(name_save[i],att_pred_t_save[i],att_pred_adv_save[i]),fmt='%s')
##Saving top highest value attributes for clean images##
with open('highest_attr_all.txt','wb') as f:
    for i in  (range(len(name_save_high))):

        np.savetxt (f,zip(name_save_high[i],att_pred_high_save[i]),fmt='%s')

##Saving top highest value attributes for adv images##
with open('highest_attr_all_adv.txt','wb') as f:
    for i in  (range(len(name_save_high))):

        np.savetxt (f,zip(name_save_high_adv[i],att_pred_high_adv_save[i]),fmt='%s')
##Saving correct and wrong classes##
with open('discriminative_attr_class_all.txt','wb') as f:
    np.savetxt (f,zip(class_save,class_adv_save),fmt='%s')


## Attribute selection for grounding ###
scale=25
name_save_clean=[]
index_save_clean=[]
name_save_adv=[]
index_save_adv=[]
att_ground_clean=[]
att_ground_adv=[]
SJE_INCORRECT_LABEL_ADV=[]
SJE_CORRECT_LABEL=[]
for j in incorrect_index:
    SJE_INCORRECT_LABEL_ADV.append(int(SJE_incorrect_label_adv[j]))
for j in correct_index:
    SJE_CORRECT_LABEL.append(int(SJE_correct_label[j]))
for i in range (len(SJE_INCORRECT_LABEL_ADV)):
    name_save_adv.append(att_names[np.argpartition((ATT_PRED_ADV[i]-ATTRIBUTES[SJE_INCORRECT_LABEL_ADV[i]]),scale)[:scale]])
    index_save_adv.append(att_index[np.argpartition((ATT_PRED_ADV[i]-ATTRIBUTES[SJE_INCORRECT_LABEL_ADV[i]]),scale)[:scale]]-1)
    att_ground_adv.append(ATT_PRED_ADV[i][att_index[np.argpartition((ATT_PRED_ADV[i]-ATTRIBUTES[SJE_INCORRECT_LABEL_ADV[i]]),scale)[:scale]]-1])
    name_save_clean.append(att_names[np.argpartition((ATT_PRED_T[i]-ATTRIBUTES[SJE_CORRECT_LABEL[i]]),scale)[:scale]])
    index_save_clean.append(att_index[np.argpartition((ATT_PRED_T[i]-ATTRIBUTES[SJE_CORRECT_LABEL[i]]),scale)[:scale]]-1)  ##-1 from index because att index is starting from 1 instead of 0
    att_ground_clean.append(ATT_PRED_T[i][att_index[np.argpartition((ATT_PRED_T[i]-ATTRIBUTES[SJE_CORRECT_LABEL[i]]),scale)[:scale]]-1])

##Saving attributes for grounding on clean images##
with open('ground_clean_attr_25.txt','wb') as f:
    for i in  (range(len(name_save_clean))):

        np.savetxt (f,zip(name_save_clean[i],att_ground_clean[i]),fmt='%s')

##Saving attributes for grounding on adv images##
with open('ground_adv_attr_25.txt','wb') as f:
    for i in  (range(len(name_save_adv))):

        np.savetxt (f,zip(name_save_adv[i],att_ground_adv[i]),fmt='%s')

############
"""
imgs_test = np.zeros((5794,3,224,224))
labels_list=[]
with open('out_test.txt') as inputfile:
    for row in csv.reader(inputfile):
        labels_list.append(row[0])
inputfile.closed

i=0
for  data in (test_loader):

    images,labels_test =data
    #print (images.shape)
    imgs_test[i:i+images.shape[0],:,:,:]=(images)

    i=i+batch_size

imgs_test_adv = np.zeros((5794,3,224,224))

i=0
for  data in (test_loader_adv):

    images,labels_test =data
#   print (images.shape)
    imgs_test_adv[i:i+images.shape[0],:,:,:]=(images)

    i=i+batch_size


for j in correct_image_no:
#     print(imgs_test.index(imgs_test[SJE_correct_index[j]]))
    print (j)
    plt.figure()
    #plt.grid(False)
    plt.axis("off")
    #print (SJE_correct_index[j])
    plt.imshow(toimage(imgs_test[j]))
    plt.savefig('clean_images/clean{0}.jpg'.format(j),dpi=400,bbox_inches='tight')

for j in incorrect_image_no:
#     print(imgs_test.index(imgs_test[SJE_correct_index[j]]))
    print (j)
    plt.figure()
    #plt.grid(False)
    plt.axis("off")
    #print (SJE_correct_index[j])
    plt.imshow(toimage(imgs_test_adv[j]))
    plt.savefig('adv_images/Adv{0}.jpg'.format(j),dpi=400,bbox_inches='tight')
###########
"""
## Euclidean Distance & similarity measurement
#1.  $\parallel W\theta(x)-W\theta(x^{adv})\parallel$  & $\parallel \phi(y^{target})-\phi(y^{adv})\parallel$
#2.  $\parallel W\theta(x^{adv})-W\theta(x)\parallel$, Where $x \in y^{adv}$  & $\parallel W\theta(x^{adv})-W\theta(x)\parallel$, Where $x \in y^{target}$
#3.  $\parallel W\theta(x^{adv})-\phi(y^{adv})\parallel$  &  $\parallel W\theta(x^{adv})-\phi(y^{target})\parallel$



diff_temp=[]
diff_ground_temp=[]
counter=0
diff_ground_temp_11=[]
dist_1=[]
dist_2=[]
dist_3=[]
dist_4=[]
dist_5=[]
dist_6=[]
dist_7=[]
att_index_pred=[]
dist_8=[]
for j in range(len(correct_labels)):

    print ("########### Eucledian distance between adversarial image attribute and clean image attribute ###########")
    diff_temp=distance.euclidean(ATT_PRED_T[j][np.argpartition((ATT_PRED_T[j]-ATT_PRED_ADV[j]),-scale)[-scale:]],ATT_PRED_ADV[j][np.argpartition((ATT_PRED_T[j]-ATT_PRED_ADV[j]),-scale)[-scale:]])
    simi_temp=pairwise.cosine_similarity(ATT_PRED_T[j][np.argpartition((ATT_PRED_T[j]-ATT_PRED_ADV[j]),-scale)[-scale:]].reshape(-1, 1) ,ATT_PRED_ADV[j][np.argpartition((ATT_PRED_T[j]-ATT_PRED_ADV[j]),-scale)[-scale:]].reshape(-1, 1) )
    print (diff_temp)
    dist_3.append(diff_temp)
    print ("########### Eucledian distance between adversarial class attribute and clean class attribute ###########")
    diff_ground_temp=distance.euclidean(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])][np.argpartition(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]-ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])],-scale)[-scale:]],ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])][np.argpartition(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]-ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])],-scale)[-scale:]])
    simi_ground_temp=pairwise.cosine_similarity(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])][np.argpartition(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]-ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])],-scale)[-scale:]].reshape(-1, 1) ,ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])][np.argpartition(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]-ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])],-scale)[-scale:]].reshape(-1, 1) )
    print (diff_ground_temp)
    dist_4.append(diff_ground_temp)
    if (int(SJE_CORRECT_LABEL[j])!=int(SJE_INCORRECT_LABEL_ADV[j])):
        print ("########### Eucledian distance between adversarial image attribute and incorrect clean image attribute ###########")

        a= (distance.euclidean(ATT_PRED_T[int(SJE_INCORRECT_LABEL_ADV[j])][np.argpartition((ATT_PRED_T[int(SJE_INCORRECT_LABEL_ADV[j])]-ATT_PRED_ADV[j]),-scale)[-scale:]],ATT_PRED_ADV[j][np.argpartition((ATT_PRED_T[int(SJE_INCORRECT_LABEL_ADV[j])]-ATT_PRED_ADV[j]),-scale)[-scale:]]))
        print(int(SJE_INCORRECT_LABEL_ADV[j]))
        print (a)
        dist_5.append(a)
        print ("########### Eucledian distance between adversarial image attribute and correct clean image attribute ###########")

        b= (distance.euclidean(ATT_PRED_T[int(SJE_CORRECT_LABEL[j])][np.argpartition((ATT_PRED_T[int(SJE_CORRECT_LABEL[j])]-ATT_PRED_ADV[j]),-scale)[-scale:]],ATT_PRED_ADV[j][np.argpartition((ATT_PRED_T[int(SJE_CORRECT_LABEL[j])]-ATT_PRED_ADV[j]),-scale)[-scale:]]))
        dist_6.append(b)
        print(int(SJE_CORRECT_LABEL[j]))
        print (b)
    print ("Distance between adversarial image attribute and incorrect class attribute")
    diff_ground_temp_1=distance.euclidean(ATT_PRED_ADV[j][np.argpartition((ATT_PRED_T[j]-ATT_PRED_ADV[j]),-scale)[-scale:]],ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])][np.argpartition(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]-ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])],-scale)[-scale:]])
    print (diff_ground_temp_1)
    diff_ground_temp_11.append(diff_ground_temp_1)
    print ("Distance between adversarial image attribute and correct class attribute")
    diff_ground_temp_2=distance.euclidean(ATT_PRED_ADV[j][np.argpartition((ATT_PRED_T[j]-ATT_PRED_ADV[j]),-scale)[-scale:]],ATTRIBUTES[int(SJE_CORRECT_LABEL[j])][np.argpartition(ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]-ATTRIBUTES[int(SJE_INCORRECT_LABEL_ADV[j])],-scale)[-scale:]])
    print (diff_ground_temp_2)
    dist_1.append(diff_ground_temp_1)
    dist_2.append(diff_ground_temp_2)
    att_index_pred.append(att_index[np.argpartition((ATT_PRED_T[j]-ATT_PRED_ADV[j]),-scale)[-scale:]])
    print ("#Eucledian distance between correct class  attribute and correct class clean image attribute#")
    d=distance.euclidean(ATT_PRED_T[j][np.argpartition((ATT_PRED_T[j]-ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]),-scale)[-scale:]],ATTRIBUTES[int(SJE_CORRECT_LABEL[j])][np.argpartition((ATT_PRED_T[j]-ATTRIBUTES[int(SJE_CORRECT_LABEL[j])]),-scale)[-scale:]])
    dist_8.append(d)

for j in range(len(incorrect_index_AT)):
    dist_7.append(distance.euclidean(ATT_PRED_ADV_I[j][np.argpartition((ATT_PRED_ADV_I[j]-ATT_PRED_ADV_AT_I[j]),-scale)[-scale:]],ATT_PRED_ADV_AT_I[j][np.argpartition((ATT_PRED_ADV_I[j]-ATT_PRED_ADV_AT_I[j]),-scale)[-scale:]]))
att_index_pred=np.asarray(att_index_pred)
plt.figure()

# plt.plot(y,dist_1,label="Distance b/w predicted Adv attr & Ground truth incorrect class attr")
# plt.plot(y,dist_2,label="Distance b/w predicted Adv attr & Ground truth correct class attr")
fig, ax = plt.subplots( nrows=1, ncols=1 )    # The big subplot
ax.hist(dist_1,label="d1") #Distance b/w pred Adv attr & Ground truth incorrect class attr
ax.hist(dist_2,label="d2", alpha=0.5)#Distance b/w pred Adv attr & Target class attr
plt.ylabel("Frequency")
plt.xlabel("Distances")
plt.legend(loc="best")
# fig.savefig("plot3_class.png")
plt.title("\n".join(wrap("Dist b/w predicted adv attr and its GT attr , Dist b/w pred pred clean attr and its GT attr")))
fig.savefig("plots/plot3.jpg")

fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
ax1.hist(dist_5,label="d1") #Eucledian distance between adversarial image attribute and incorrect clean image attribute
ax1.hist(dist_6,label="d2", alpha=0.5) #Eucledian distance between adversarial image attribute and correct clean image attribute
plt.ylabel("Frequency")
plt.xlabel("Distances")
plt.legend(loc="best")
plt.title("\n".join(wrap("Dist b/w predicted adv attr and predicted clean attr from same class, Dist b/w pred clean and pred adv attr")))
fig1.savefig("plots/plot2.jpg")

fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
ax1.hist(dist_3,label="d1") #Distance b/w  pred Adv attr & pred clean image attr
ax1.hist(dist_4,label="d2", alpha=0.5) #Distance b/w Ground truth Adv class attr & Ground truth target class attr
plt.ylabel("Frequency")
plt.xlabel("Distances")
plt.legend(loc="best")
plt.title("\n".join(wrap("Dist b/w predicted clean and predicted adv attr, Dist b/w GT clean and GT adv attr")))
fig1.savefig("plots/plot1.jpg")

fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
ax1.hist(dist_3,label="d1") #Distance b/w  pred Adv attr & pred clean image attr
ax1.hist(dist_7,label="d2", alpha=0.5) #Distance b/w Ground truth Adv class attr & Ground truth target class attr
plt.ylabel("Frequency")
plt.xlabel("Distances")
plt.legend(loc="best")
plt.title("\n".join(wrap("Dist b/w predicted clean and predicted adv attr, Dist b/w wrongly classified images with AT")))
fig1.savefig("plots/plot4.jpg")

fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
ax1.hist(dist_8,label="d1") #Distance b/w  pred Adv attr & pred clean image attr
ax1.hist(diff_ground_temp_11,label="d2", alpha=0.5) #Distance b/w Ground truth Adv class attr & Ground truth target class attr
plt.ylabel("Frequency")
plt.xlabel("Distances")
plt.legend(loc="best")
fig1.savefig("plots/plot5.jpg")
#scale=2
#fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
#ax1.hist((att_pred_t[:,np.argpartition((att_pred_t[j]-att_pred_adv[j]),-scale)[-scale:]]),label="d1") #Distance b/w  pred Adv attr & pred clean image attr
#ax1.hist((att_pred_adv[:,np.argpartition((att_pred_t[j]-att_pred_adv[j]),-scale)[-scale:]]),label="d2") #Distance b/w Ground truth Adv class attr & Ground truth target class attr
#plt.ylabel("Frequency")
#plt.xlabel("Attributes")
#plt.legend(loc="best")
#plt.title("\n".join(wrap("Dist b/w predicted clean and predicted adv attr, Dist b/w wrongly classified images with AT")))
#fig1.savefig("plots/plot5.jpg")
