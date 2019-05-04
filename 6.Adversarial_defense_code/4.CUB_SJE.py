from __future__ import division
import os
import pandas as pd
import imageio
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
#print ("test set size",test_size)
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

model=torch.load("/home/sgulshad/sadaf/CUB_experiments/pytorch-adversarial_box/models/adv_train_PGD_ep16")    
model = model.cuda()
#model = torch.nn.DataParallel(model).cuda() 
checkpoint = torch.load("/home/sgulshad/sadaf/CUB_experiments/pytorch-adversarial_box/models/adv_train_PGD_ep16.pth")        
new_state_dict = OrderedDict()
#state_dict =checkpoint['state_dict']
state_dict =checkpoint

for i in range(0, len(model.state_dict().keys())):
    new_state_dict[model.state_dict().keys()[i]] = model.state_dict()[model.state_dict().keys()[i]]

model.load_state_dict(new_state_dict,strict=False)

model.eval()
#print (model.eval())
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')


##Training image feature extraction
with open('out.txt', 'w') as f:
	train_images=[]
    # for phase in ['train_split']:
	for data in (train_loader):
#         get the inputs
#         print (i)
#             
		images,labels_train =data
		
  #   	print (labels_train.shape)         
		if use_gpu:
			images = Variable(images.cuda())
#       
		else:
			images, _ = Variable(images)
		# if (int(images.shape[0])==batch_size):
		# 	# print(images.shape[0])
		# 	my_embedding = torch.zeros(batch_size,2048)
		# else:
		# 	my_embedding = torch.zeros(3,2048)
		my_embedding = torch.zeros(int(images.shape[0]),2048)
        # 4. Define a function that will copy the output of a layer
		def copy_data(m, i, o):
#             print (o.data)
			my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        
		h = layer.register_forward_hook(copy_data)

        # 6. Run the model on our transformed image
		model(images)

        # 7. Detach our copy function from the layer
		h.remove()
		imgs_train_fea=my_embedding.numpy()
#       print (imgs_train_fea.shape)
		labels_num_train=labels_train.cpu()
		labels_num_train=labels_num_train.numpy()
            
		train_images.append(imgs_train_fea)
#       print (type(labels_train))
		np.savetxt(f, labels_num_train,fmt='% 4d')

train_images=np.asarray(train_images)
print (train_images.shape)
# train_images=np.reshape(train_images,(-1,2048))
train_images=np.vstack(train_images)

print ("train_images", train_images.shape)

##Validation image feature extraction
with open('out_val.txt', 'w') as f:
	val_images=[]
	# for phase in ['train_split']:
	for data in (val_loader):
#         get the inputs
#         print (i)
#             
		images,labels_val =data
		
  #   	print (labels_train.shape)         
		if use_gpu:
			images = Variable(images.cuda())
#       
		else:
			images, _ = Variable(images)
		# if (int(images.shape[0])==batch_size):
		# 	# print(images.shape[0])
		# 	my_embedding = torch.zeros(batch_size,2048)
		# else:
		# 	my_embedding = torch.zeros(3,2048)
		my_embedding = torch.zeros(int(images.shape[0]),2048)
        # 4. Define a function that will copy the output of a layer
		def copy_data(m, i, o):
#             print (o.data)
			my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        
		h = layer.register_forward_hook(copy_data)

        # 6. Run the model on our transformed image
		model(images)

        # 7. Detach our copy function from the layer
		h.remove()
		imgs_val_fea=my_embedding.numpy()
#       print (imgs_train_fea.shape)
		labels_num_val=labels_val.cpu()
		labels_num_val=labels_num_val.numpy()
            
		val_images.append(imgs_val_fea)
#       print (type(labels_train))
		np.savetxt(f, labels_num_val,fmt='% 4d')
        

val_images=np.asarray(val_images)
val_images=np.vstack(val_images)
print ("val_images",(val_images).shape)
# val_images=np.reshape(val_images,(-1,512))
## Test image feature extraction
with open("out_test.txt","w") as f:
    from sklearn import metrics
    test_images=[]

    for data in (test_loader):
#         get the inputs
#         print (i)
#             
		images,labels_test =data
		
  #   	print (labels_train.shape)         
		if use_gpu:
			images = Variable(images.cuda())
#       
		else:
			images, _ = Variable(images)
		# if (int(images.shape[0])==batch_size):
		# 	# print(images.shape[0])
		# 	my_embedding = torch.zeros(batch_size,2048)
		# else:
		# 	my_embedding = torch.zeros(3,2048)
		my_embedding = torch.zeros(int(images.shape[0]),2048)
        # 4. Define a function that will copy the output of a layer
		def copy_data(m, i, o):
#             print (o.data)
			my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        
		h = layer.register_forward_hook(copy_data)

        # 6. Run the model on our transformed image
		model(images)


        # Detach our copy function from the layer
		h.remove()
		imgs_test_fea_clean=my_embedding.numpy()
		labels_num_test=labels_test.cpu()
		labels_num_test=labels_num_test.numpy()
		test_images.append(imgs_test_fea_clean)

		np.savetxt(f,labels_num_test,fmt='% 4d')
test_images=np.asarray(test_images)
test_images=np.vstack(test_images)
print ("test_images",(test_images).shape)

## Adv Test image feature extraction
with open("out_test_adv.txt","w") as f:
    from sklearn import metrics
    test_images_adv=[]
    for data in (test_loader_adv):
#         get the inputs
#         print (i)
#             
		images,labels_test_adv =data
		
  #   	print (labels_train.shape)         
		if use_gpu:
			images = Variable(images.cuda())
#       
		else:
			images, _ = Variable(images)
		# if (int(images.shape[0])==batch_size):
		# 	# print(images.shape[0])
		# 	my_embedding = torch.zeros(batch_size,2048)
		# else:
		# 	my_embedding = torch.zeros(3,2048)
		my_embedding = torch.zeros(int(images.shape[0]),2048)
        # 4. Define a function that will copy the output of a layer
		def copy_data(m, i, o):
#             print (o.data)
			my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        
		h = layer.register_forward_hook(copy_data)

        # 6. Run the model on our transformed image
		model(images)


        # Detach our copy function from the layer
		h.remove()
		imgs_test_fea_adv=my_embedding.numpy()
		labels_num_test_adv=labels_test_adv.cpu()
		labels_num_test_adv=labels_num_test_adv.numpy()
		test_images_adv.append(imgs_test_fea_adv)

		np.savetxt(f,labels_num_test_adv,fmt='% 4d')
test_images_adv=np.asarray(test_images_adv)
test_images_adv=np.vstack(test_images_adv)
print ("test_images_adv",(test_images_adv).shape)


##Training SJE
## Code for SJE
def argmaxOverMatrices(x, y, W):
	projected_x = np.dot(x ,W)
	score = np.dot(projected_x , y)     
	return(score)

n_train = train_images.shape[0]
n_class = ATTRIBUTES.T.shape[1]
print (n_class)
X=train_images
Y=ATTRIBUTES.T
labels_list=[]
eta = 1e-2
with open('out.txt') as inputfile:
	for row in csv.reader(inputfile):
		labels_list.append(row[0])
inputfile.closed
labels=np.array(labels_list)
labels = [int(x) for x in labels]
labels=np.asarray(labels)

## Initialization
W= 1.0/np.sqrt(X.shape[1]) * np.random.rand(X.shape[1], Y.shape[0])
n_epoch=100
max_idx_array=np.zeros((X.shape[0],1))
W_best = np.copy(W)
best_accuracy = 0
##SGD
for e in range(0,n_epoch):
	print (e)
	number_of_true = 0
	max_idx_list=[]
	perm = np.random.permutation(n_train)
	for i in range (0,n_train):
		ni = perm[i]
		scores=argmaxOverMatrices(X[ni,:], Y, W)
		scores[labels[ni]]=scores[labels[ni]]-1
		max_idx=np.argmax(scores)

		if(max_idx!=labels[ni]):
			W = W + eta * np.dot(X[ni,:].reshape(len(X[ni,:]),1),(Y[:,labels[ni]] - Y[:,max_idx].T).reshape(len(Y[:,labels[ni]]),1).T)
		elif (max_idx) == labels[ni]:
			number_of_true += 1
	print ("Number of true " + str(number_of_true))

	X_train=train_images
	scores_train = np.dot(np.dot(X_train , W ), Y)

	predict_label_train = np.argmax(scores_train,axis=1)
	labels_list=[int(x) for x in labels_list ]
	count_train=0
	for i in range (len(labels)):
		if predict_label_train[i]==labels_list[i]:
			count_train=count_train+1
	print ("count_train", count_train)
	print ("label length",len(labels))
	train_acc=(count_train/len(labels))*100
	print ("Train acc",train_acc)
	pred_attr_train=(np.dot(X_train , W ))
	pred_attr_train=normalize(pred_attr_train)
	####################################
    ##validation
	X_val=val_images
    # print(labels_list)
	pred_attr_val=(np.dot(X_val , W ))
	pred_attr_val=normalize(pred_attr_val)
	scores_val = np.dot(np.dot(X_val , W ), Y)
	predict_label_val = np.argmax(scores_val,axis=1)
	labels_list_val=[]
	with open('out_val.txt') as inputfile:
		for row in csv.reader(inputfile):
			labels_list_val.append(row[0])
	inputfile.closed
	labels_list_val=[int(x) for x in labels_list_val ]
	count_val=0
	for i in range (len(labels_list_val)):
		if predict_label_val[i]==labels_list_val[i]:
			count_val=count_val+1
	accuracy=(count_val/len(labels_list_val))*100
	print ("Val acc",(count_val/len(labels_list_val))*100)
	if accuracy > best_accuracy :
		best_accuracy = accuracy
		W_best = np.copy(W)
	print ("Best accuracy so far is :" + str(best_accuracy))

with open('out_attr_train.txt','wb') as f:
    for i in  ((pred_attr_train)):
        np.savetxt (f,i,fmt='%.6f')
with open('out_attr_val.txt','wb') as f:
    for i in  ((pred_attr_val)):
        np.savetxt (f,i,fmt='%.6f')


##Testing SJE
labels_list=[]
SJE_correct_index=[]
SJE_correct_label=[]
SJE_incorrect_index=[]
SJE_incorrect_label=[]
with open('out_test.txt') as inputfile:
    for row in csv.reader(inputfile):
        labels_list.append(row[0])
inputfile.closed
labels_test=np.array(labels_list)
labels_test = [int(x) for x in labels_test]
labels_test=np.asarray(labels_test)
n_samples = len(labels_test)
n_class = len(np.unique(labels_test))
print(n_samples)
print (n_class)
X=test_images
W=W_best
pred_attr=(np.dot(X , W ))
pred_attr=normalize(pred_attr)
scores = np.dot(np.dot(X , W ), Y)
print (scores.shape)
predict_label = np.argmax(scores,axis=1)
print (predict_label)

count=0
for i in range (len(labels_test)):
    if predict_label[i]==labels_test[i]:
        count=count+1
        SJE_correct_index.append(i)
        SJE_correct_label.append(predict_label[i])
    else:
        SJE_incorrect_index.append(i)
        SJE_incorrect_label.append(predict_label[i])
print ("Test accuracy",(count/len(labels_test))*100)
with open('out_attr_test.txt','wb') as f:
    for i in  ((pred_attr)):
        np.savetxt (f,i,fmt='%.6f')

W_train=W
##predicted classes
pred_class=predict_label
pred_class=np.asarray(pred_class)
print (pred_class.shape)
##ground truth classes
test_classes=[]
with open('out_test.txt') as inputfile:
    for row in csv.reader(inputfile):
        test_classes.append(row[0])
inputfile.closed
test_classes=np.array(test_classes)
test_classes = [int(x) for x in test_classes]
correct_index=[]
for i in range (pred_class.shape[0]):
    if (pred_class[i]==test_classes[i]):
        correct_index.append(i)
correct_index=np.asarray(correct_index)
# print (len(correct_index))
# print (correct_index[1])


with open("SJE_correct_index.txt","w") as f:
    np.savetxt(f,zip(SJE_correct_index,SJE_correct_label), fmt="%d")
with open("SJE_incorrect_index.txt","w") as f:
    np.savetxt(f,zip(SJE_incorrect_index,SJE_incorrect_label), fmt="%d")


##Testing SJE for adversarial images
labels_list=[]
SJE_correct_index=[]
SJE_correct_label=[]
SJE_incorrect_index=[]
SJE_incorrect_label=[]
with open('out_test_adv.txt') as inputfile:
	for row in csv.reader(inputfile):
		labels_list.append(row[0])
inputfile.closed
labels_test=np.array(labels_list)
labels_test = [int(x) for x in labels_test]
labels_test=np.asarray(labels_test)
n_samples = len(labels_test)
n_class = len(np.unique(labels_test))
print(len(labels_test))
print (n_class)
X=test_images_adv
W=W_best
pred_attr=(np.dot(X , W ))
pred_attr=normalize(pred_attr)
print (pred_attr[0][209])
# pred_attr[0][209]=27.264119
Y=ATTRIBUTES.T
scores = np.dot(pred_attr, Y)
print (scores.shape)
predict_label = np.argmax(scores,axis=1)
print (predict_label)

count=0
for i in range (len(labels_test)):
	if predict_label[i]==labels_test[i]:
		count=count+1
		SJE_correct_index.append(i)
		SJE_correct_label.append(predict_label[i])
	else:
		SJE_incorrect_index.append(i)
		SJE_incorrect_label.append(predict_label[i])
print ("count",count)        
print ("Test accuracy for adversarial images",(count/len(labels_test))*100)
with open('out_attr_test_adv.txt','wb') as f:
    for i in  ((pred_attr)):
        np.savetxt (f,i,fmt='%.6f')

with open("SJE_correct_index_adv.txt","w") as f:
    np.savetxt(f,zip(SJE_correct_index,SJE_correct_label), fmt="%d")
with open("SJE_incorrect_index_adv.txt","w") as f:
    np.savetxt(f,zip(SJE_incorrect_index,SJE_incorrect_label), fmt="%d")

att_pred_t=np.loadtxt('out_attr_test.txt')
att_pred_t=att_pred_t.reshape(-1,312)
att_pred_adv=np.loadtxt('out_attr_test_adv.txt')
att_pred_adv=att_pred_adv.reshape(-1,312)
