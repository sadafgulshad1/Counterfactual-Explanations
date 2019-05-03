import cv2
import csv
import torch
import torchvision
from torchvision import datasets, models, transforms

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
#from easydict import EasyDict
import os

# set display defaults
plt.rcParams['figure.figsize'] = (25, 9)        # small images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# Change dir to caffe root or prototxt database paths won't work wrong
import os
print os.getcwd()
os.chdir('..')
print os.getcwd()

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

import caffe

data_path = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/data/genome/1600-400-20'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())

# Check object extraction
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
import cv2

GPU_ID = 0   # if we have multiple GPUs, pick one 
caffe.set_device(GPU_ID)  
caffe.set_mode_gpu()
net = None
cfg_from_file('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml')

weights = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)


# print ((attributes))

att_names=[]
att_index=[]
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/ground_clean_attr.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names.append(row[0].split('::')[1].split(" ")[0])
        att_index.append(row[0].split(' ')[0])
inputfile.closed
att_names=np.array(att_names)
att_index=[int(x) for x in att_index]
att_index= (np.asarray(att_index))
for i in range (len(att_names)):
    print((att_names[i*10:(i*10+10)]))

# print ((attributes))
att_names_adv=[]
att_index_adv=[]
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/ground_adv_attr.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names_adv.append(row[0].split('::')[1].split(" ")[0])
        att_index_adv.append(row[0].split(' ')[0])
inputfile.closed
att_names_adv=np.array(att_names_adv)
att_index_adv=[int(x) for x in att_index_adv]
att_index_adv= (np.asarray(att_index_adv))
for i in range (len(att_names_adv)):
    print((att_names_adv[i*10:(i*10+10)]))

data_transforms = {
    'CUB_clean': transforms.Compose([
                                     transforms.Scale(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
#                                      transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    ,
    'CUB_adv': transforms.Compose([
                                     transforms.Scale(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
#                                      transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ]),
}
data_dir = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['CUB_clean', 'CUB_adv']}
#print (image_datasets)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=False, num_workers=4)
              for x in ['CUB_clean', 'CUB_adv']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['CUB_clean', 'CUB_adv']}
class_names = image_datasets['CUB_clean'].classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes1 = next(iter(dataloaders['CUB_clean']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes1])
use_gpu = torch.cuda.is_available()


im_file = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/CUB_clean/CUB_clean/clean0.jpg'
#im_file="/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/antelope_10140.jpg"
#im_file="/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Tensorflow_code/robust-physical-attack/perturbation.png"
###########################
# Similar to get_detections_from_im
conf_thresh=0.7
min_boxes=5
max_boxes=5

im = cv2.imread(im_file)
scores, boxes, attr_scores, rel_scores = im_detect(net, im)

# Keep the original boxes, don't worry about the regression bbox outputs
rois = net.blobs['rois'].data.copy()
# unscale back to raw image space
blobs, im_scales = _get_blobs(im, None)

cls_boxes = rois[:, 1:5] / im_scales[0]
cls_prob = net.blobs['cls_prob'].data
attr_prob = net.blobs['attr_prob'].data
pool5 = net.blobs['pool5_flat'].data
print (cls_prob.shape)
# Keep only the best detections
max_conf = np.zeros((rois.shape[0]))
for cls_ind in range(1,cls_prob.shape[1]):
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = np.array(nms(dets, cfg.TEST.NMS))
    max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

keep_boxes = np.where(max_conf >= conf_thresh)[0]
if len(keep_boxes) < min_boxes:
    keep_boxes = np.argsort(max_conf)[::-1][:min_boxes]
elif len(keep_boxes) > max_boxes:
    keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]
############################

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)

boxes = cls_boxes[keep_boxes]
objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
print(objects)
attr_thresh = 0.1
attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)
print (attr)
for i in range(len(keep_boxes)):
    bbox = boxes[i]
    if bbox[0] == 0:
        bbox[0] = 1
    if bbox[1] == 0:
        bbox[1] = 1
    cls = classes[objects[i]+1]
    
    if attr_conf[i] > attr_thresh:
		for k in range (len(att_names[j*10:(j*10+10)])):
			if attributes[attr[i]+1]==att_names[k]:
				cls = attributes[attr[i]+1] + " " + cls
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2, alpha=0.5)
                )
                plt.gca().text(bbox[0], bbox[1] - 2,
                    '%s' % (cls),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=10, color='white')
		plt.savefig('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/clean_bb/1.jpg') 
print 'boxes=%d' % (len(keep_boxes))
"""
for  j, data in enumerate(dataloaders['CUB_clean']):
    print(j)      
    im,_ =data
    conf_thresh=0.4
    min_boxes=36
    max_boxes=36
    im =im.numpy()
    im=(np.reshape(im,(3,224,224)))
#     im=np.swapaxes(im,0,2)
    im=im[:, :, [2, 1, 0]] 
    
    #im_file='/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/Adv 0.jpg'
    
    
    
    #im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regression bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data
    # print (cls_prob.shape)
    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]
############################

#     
    #print (im.shape)
    #plt.imshow(im)

    boxes = cls_boxes[keep_boxes]
    objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    # print(objects)
    attr_thresh = 0.1
    attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)
    # print (attr)
    for i in range(len(keep_boxes)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects[i]+1]
#     print (cls)
        if attr_conf[i] > attr_thresh:

#         if attributes[attr[i]+1]=="pink":
#             print ("pink")
#         cls = attributes[attr[i]+1]
#         cls = attributes[attr[i]+1] + " " + cls
            for k in range (len(att_names[j*10:(j*10+10)])):
                if attributes[attr[i]+1]==att_names[k]:
                    cls = attributes[attr[i]+1] + " " + cls
                    plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2, alpha=0.5)
                )
                    plt.gca().text(bbox[0], bbox[1] - 2,
                    '%s' % (cls),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=10, color='white')

            plt.savefig('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/clean_bb/clean_bb{0}.jpg'.format(j)) 
            
    print 'boxes=%d' % (len(keep_boxes))
"""
