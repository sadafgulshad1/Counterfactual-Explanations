#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib
matplotlib.use('Agg')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse,glob,csv

data_path = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/data/genome/1600-400-20'
scale=25
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



def demo_tuples(net, image_name):
    """Detect objects, attributes and relations in an image using pre-computed object proposals."""
    image_num=int(image_name.split(".")[0])
    att_unique=np.unique(att_names[image_num*scale:(image_num*scale+scale)])
    print (att_unique)
    att_unique_adv=np.unique(att_names_adv[image_num*scale:(image_num*scale+scale)])
    cls_unique=np.unique(att_cls[image_num*scale:(image_num*scale+scale)])
    print (cls_unique)
    cls_unique_adv=np.unique(att_cls_adv[image_num*scale:(image_num*scale+scale)])
    # Load the demo image
    im_file = os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/CUB_clean", image_name)
    im = cv2.imread(im_file)
    print (im.shape)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)
    if attr_scores is not None:
        print 'Found attribute scores'
    """
    if rel_scores is not None:
        print 'Found relation scores'
        rel_scores = rel_scores[:,1:] # drop no relation
        rel_argmax = np.argmax(rel_scores, axis=1).reshape((boxes.shape[0],boxes.shape[0]))
        rel_score = np.max(rel_scores, axis=1).reshape((boxes.shape[0],boxes.shape[0]))
    """    
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])    

    # Visualize detections for each class
    CONF_THRESH = 0.2
    NMS_THRESH = 0.05
    ATTR_THRESH = 0.1
    
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im)
    
    # Detections
    det_indices = []
    det_scores = []
    det_objects = []
    det_bboxes = []
    det_attrs = []
    
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, NMS_THRESH))
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        
        if len(inds) > 0:
            keep = keep[inds]
            for k in keep:
                det_indices.append(k)
                det_bboxes.append(cls_boxes[k])
                det_scores.append(cls_scores[k])
                det_objects.append(cls)
                if attr_scores is not None:
                    attr_inds = np.where(attr_scores[k][1:] >= ATTR_THRESH)[0]
                    det_attrs.append([attributes[ix] for ix in attr_inds])
                else:
                    det_attrs.append([])
            #det_attrs=[element for element in det_attrs if element in att_unique]
            det_objects1=[element for element in det_objects if element in cls_unique]
            det_objects_index=[det_objects.index(element) for element in det_objects if element in cls_unique]

    #rel_score = rel_score[det_indices].T[det_indices].T
    #rel_argmax = rel_argmax[det_indices].T[det_indices].T
    for i,(idx,score,obj,bbox,attr) in enumerate(zip(det_indices,det_scores,det_objects1,det_bboxes,det_attrs)):
        print (idx)
        attr=[element for element in attr if element in att_unique]
        
            
            
        #if obj in cls_unique:
        box_text=obj
        if len(attr) > 0:
            box_text + " " +(attr)
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=2, alpha=0.5))
        ax.text(bbox[0], bbox[1] - 2,'%s' % (box_text),bbox=dict(facecolor='blue', alpha=0.5),fontsize=10, color='white')

    plt.axis('off')
    plt.tight_layout()
    #plt.draw()    
    plt.savefig(('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/clean_bb/'+image_name))    
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
    
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        
    args = parse_args()
    net = None
    cfg_from_file('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml')

    caffemodel = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    prototxt = '/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/bottom-up-attention/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    
    file_names=[glob.glob("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/CUB_clean/*.jpg")]
    im_names=[]
    image_no=[]
    #import zip
    for i in range (len(file_names[0])):
        im_names.append(file_names[0][i].split("/")[9])
        image_no.append(i)

    #print (im_names)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo_tuples(net, im_name)

    
