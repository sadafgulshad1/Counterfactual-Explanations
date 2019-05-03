# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
# display plots in this notebook
#%matplotlib inline
from itertools import groupby
import argparse
import os, sys,csv,glob,random
#import torch
from shutil import copy
print (sys.executable)
# set display defaults
plt.rcParams['figure.figsize'] = (12, 9)        # small images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


# Change dir to caffe root or prototxt database paths won't work wrong
import os
print os.getcwd()
os.chdir('..')
print os.getcwd()
scale=25
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

import caffe

data_path = './data/genome/1600-400-20'

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
def parse_args():
    """Parse input arguments."""
    
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    args = parser.parse_args()

    return args
args = parse_args()
if args.cpu_mode:
   caffe.set_mode_cpu()
else:
   caffe.set_mode_gpu()
   caffe.set_device(args.gpu_id)
   cfg.GPU_ID = args.gpu_id
    #net = caffe.Net(prototxt, caffemodel, caffe.TEST)

#GPU_ID = 0   # if we have multiple GPUs, pick one 
#caffe.set_device(GPU_ID)  
#caffe.set_mode_gpu()
net = None
cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)

##############################################
#Loading attributes
att_names=[]
att_index=[]
att_cls=[]
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/ground_clean_attr_25.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names.append(row[0].split('::')[1].split(" ")[0])
        att_cls.append(row[0].split("_")[1].split("_")[0].split(" ")[0])
        att_index.append(row[0].split(' ')[0])
inputfile.closed
att_cls = [word.replace('bill','beak') for word in att_cls]
att_names = [word.replace('curved_(up_or_down)','curved') for word in att_names]
att_names = [word.replace('grey','gray') for word in att_names]
att_names = [word.replace('multi-colored','multi colored') for word in att_names]
att_names = [word.replace('rounded_tail','round') for word in att_names]
att_names = [word.replace('pointed_tail','pointed') for word in att_names]
att_names = [word.replace('squared_tail','square') for word in att_names]
att_names = [word.replace('longer_than_head','long') for word in att_names]
att_names = [word.replace('shorter_than_head','short') for word in att_names]
att_names = [word.replace('long-wings','long') for word in att_names]
att_names = [word.replace('pointed-wings','pointed') for word in att_names]
att_names = [word.replace('rounded-wings','round') for word in att_names]
att_names = [word.replace('large_(16_-_32_in)','large') for word in att_names]
att_names = [word.replace('very_large_(32_-_72_in)','large') for word in att_names]
att_names = [word.replace('small_(5_-_9_in)','small') for word in att_names]
att_names = [word.replace('very_small_(3_-_5_in)','small') for word in att_names]
att_names = [word.replace('upright-perching_water-like','perched') for word in att_names]
att_names = [word.replace('perching-like','perched') for word in att_names]
att_cls = [word.replace('under','tail') for word in att_cls]
att_cls = [word.replace('upper','tail') for word in att_cls]
att_names=np.array(att_names)

#print (att_cls)
att_cls=np.array(att_cls)


att_index=[int(x) for x in att_index]
att_index= (np.asarray(att_index))
for i in range (len(att_names)/4011):
    print((att_names[i*scale:(i*scale+scale)]))

# print ((attributes))
att_names_adv=[]
att_index_adv=[]
att_cls_adv=[]
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/tools/ground_adv_attr_25.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names_adv.append(row[0].split('::')[1].split(" ")[0])
        att_cls_adv.append(row[0].split('_')[1].split('_')[0])
        att_index_adv.append(row[0].split(' ')[0])
inputfile.closed
att_cls_adv = [word.replace('bill','beak') for word in att_cls_adv]
att_names_adv = [word.replace('curved_(up_or_down)','curved') for word in att_names_adv]
att_names_adv = [word.replace('grey','gray') for word in att_names_adv]
att_names_adv = [word.replace('multi-colored','multi colored') for word in att_names_adv]
att_names_adv = [word.replace('rounded_tail','round') for word in att_names_adv]
att_names_adv = [word.replace('pointed_tail','pointed') for word in att_names_adv]
att_names_adv = [word.replace('squared_tail','square') for word in att_names_adv]
att_names_adv = [word.replace('longer_than_head','long') for word in att_names_adv]
att_names_adv = [word.replace('shorter_than_head','short') for word in att_names_adv]
att_names_adv = [word.replace('long-wings','long') for word in att_names_adv]
att_names_adv = [word.replace('pointed-wings','pointed') for word in att_names_adv]
att_names_adv = [word.replace('rounded-wings','round') for word in att_names_adv]
att_names_adv = [word.replace('large_(16_-_32_in)','large') for word in att_names_adv]
att_names_adv = [word.replace('very_large_(32_-_72_in)','large') for word in att_names_adv]
att_names_adv = [word.replace('small_(5_-_9_in)','small') for word in att_names_adv]
att_names_adv = [word.replace('very_small_(3_-_5_in)','small') for word in att_names_adv]
att_names_adv = [word.replace('upright-perching_water-like','perched') for word in att_names_adv]
att_names_adv = [word.replace('perching-like','perched') for word in att_names_adv]
att_cls_adv = [word.replace('under','tail') for word in att_cls_adv]
att_cls_adv = [word.replace('upper','tail') for word in att_cls_adv]
att_names_adv=np.array(att_names_adv)
att_cls_adv=np.array(att_cls_adv)
att_index_adv=[int(x) for x in att_index_adv]
att_index_adv= (np.asarray(att_index_adv))
for i in range (len(att_names_adv)/4011):
    print((att_names_adv[i*scale:(i*scale+scale)]))

#im_file = 'data/demo/004545.jpg'
#im_file="/home/sadaf/bottom-up-attention/tools/Images/Adv3957.jpg"
###########################

"""
n=1
dir_input_adv="/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/CUB_adv"
dir_input="/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/CUB_clean"
src_files = (os.listdir(dir_input))
src_files_adv = (os.listdir(dir_input_adv))
def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)

files = [f for f in src_files if valid_path(dir_input, f)]
choices = random.sample(files, n)
for i in range (len(choices)):
    indexes.append(files.index(choices[i]))
print (indexes)
dst_adv="/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/images_temp_adv"
dst="/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/images_temp"

for i in range (len(choices)):
    src=os.path.join(dir_input,choices[i])
    src_adv=os.path.join(dir_input_adv,choices[i])
    copy(src, dst)
    copy(src_adv, dst_adv)
"""
    
#print ((image_name))

#image_no= [int(''.join(i)) for j in range (len(image_name)) for is_digit, i in groupby(image_name[j].split(".")[0], str.isdigit) if is_digit]
#image_no=[int(s) for i in range (len(image_name)) for s in image_name[i].split(".")[0] if s.isdigit()]
#im = [cv2.imread(file) for file in glob.glob("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/CUB_clean/*.jpg")]
#im=np.asarray(im)

#print (im.shape)
#for j in range (im.shape[0]):
# Similar to get_detections_from_im
#fig, ax = plt.subplots()  
def demo(image_name,image_no,net):
    
    conf_thresh=0.4
    min_boxes=0
    max_boxes=15
    indexes=[]
    cfg.TEST.NMS=0.3
    
    
    im = cv2.imread(os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/CUB_clean/",image_name))
    
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regression bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    print (len(cls_boxes))
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

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
    att_unique=np.unique(att_names[image_no*scale:(image_no*scale+scale)])
    print (att_unique)
    att_unique_adv=np.unique(att_names_adv[image_no*scale:(image_no*scale+scale)])
    cls_unique=np.unique(att_cls[image_no*scale:(image_no*scale+scale)])
    print (cls_unique)
    cls_unique_adv=np.unique(att_cls_adv[image_no*scale:(image_no*scale+scale)])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(im)

    boxes = cls_boxes[keep_boxes]
    #print (boxes)
    #print (keep_boxes)
    objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    attr_thresh = 0.1
    attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)
    for i in range(len(keep_boxes)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        #cls = classes[objects[i]+1]
        if attr_conf[i] > attr_thresh:
            #for k in range (len(att_unique)):
             #   for l in range (len(cls_unique)):
                    #if attributes[attr[i]+1]==att_unique[k]:
                     #   if classes[objects[i]+1] == cls_unique[l]:
                            #if attributes[attr[i]+1] not in att_unique_adv:
                                #if classes[objects[i]+1] not in cls_unique_adv:
            if attributes[attr[i]+1] in att_unique:
                if classes[objects[i]+1] in cls_unique:
                    cls = attributes[attr[i]+1] + " " + classes[objects[i]+1]
                    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=2, alpha=0.5))
                    plt.gca().text(bbox[0], bbox[1] - 2,'%s' % (cls),bbox=dict(facecolor='blue', alpha=0.5),fontsize=10, color='white')
    
    plt.axis("off")
#plt.savefig('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/adv_bb/adv_bb{}_10_s.jpg'.format(image_no)) 
    plt.savefig('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/clean_bb/clean_bb{}_scale_s.jpg'.format(image_no)) 
    plt.close()       
                    
                    
                    
                    
                    
        #print (cls)
#print ("att_names clean")
#print (att_names[3957*10:(3957*10+10)])
#print ("att_names adv")
#print (att_names_adv[3957*10:(3957*10+10)])
file_names=[glob.glob("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/CUB_clean/*.jpg")]
image_name=[]
image_no=[]
#import zip
for i in range (len(file_names[0])):
    image_name.append(file_names[0][i].split("/")[9])
    
    image_no.append(i)
print (len(image_name))
print (len(image_no))
for j in range (len(image_no)):
    
    demo(image_name[j],image_no[j],net)
    