# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
# display plots in this notebook
#%matplotlib inline
import argparse
import os, sys,csv
import torch
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
with open('/home/sadaf/bottom-up-attention/tools/ground_clean_attr.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names.append(row[0].split('::')[1].split(" ")[0])
        att_index.append(row[0].split(' ')[0])
inputfile.closed
att_names=np.array(att_names)
att_index=[int(x) for x in att_index]
att_index= (np.asarray(att_index))
for i in range (len(att_names)):
    print((att_names[i*50:(i*50+50)]))

# print ((attributes))
att_names_adv=[]
att_index_adv=[]
with open('/home/sadaf/bottom-up-attention/tools/ground_adv_attr.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names_adv.append(row[0].split('::')[1].split(" ")[0])
        att_index_adv.append(row[0].split(' ')[0])
inputfile.closed
att_names_adv=np.array(att_names_adv)
att_index_adv=[int(x) for x in att_index_adv]
att_index_adv= (np.asarray(att_index_adv))
for i in range (len(att_names_adv)):
    print((att_names_adv[i*50:(i*50+50)]))
#im_file = 'data/demo/004545.jpg'
im_file="/home/sadaf/bottom-up-attention/tools/Images/Adv3957.jpg"
###########################
# Similar to get_detections_from_im
conf_thresh=0.4
min_boxes=36
max_boxes=36

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
att_unique=np.unique(att_names[3957*50:(3957*50+50)])
att_unique_adv=np.unique(att_names_adv[3957*50:(3957*50+50)])
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)

boxes = cls_boxes[keep_boxes]
objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
attr_thresh = 0.1
attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)
print ("fastr-rcnn attr")
for i in range(len(keep_boxes)):
    bbox = boxes[i]
    if bbox[0] == 0:
        bbox[0] = 1
    if bbox[1] == 0:
        bbox[1] = 1
    cls = classes[objects[i]+1]
    if attr_conf[i] > attr_thresh:
        
        print (attributes[attr[i]+1])
        for k in range (len(att_unique_adv)):
                
                if attributes[attr[i]+1]==att_unique_adv[k]:
                    print ("attr")
                    print (attributes[attr[i]+1])
                    
                
                    if attributes[attr[i]+1] not in att_unique:
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

        plt.savefig('/home/sadaf/bottom-up-attention/tools/adv_bb/adv_bb3957_50_s.jpg')
        #plt.savefig('/home/sadaf/bottom-up-attention/tools/clean_bb/clean_bb3957_50_s.jpg') 
           
    #print (cls)
print ("att_names clean")
print (att_names[3957*50:(3957*50+50)])
print ("att_names adv")
print (att_names_adv[3957*50:(3957*50+50)])


