# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
# display plots in this notebook
#%matplotlib inline
from itertools import groupby
import argparse
import os, sys,csv,glob,random,re
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
scale=50
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
total_no_img=6116
#GPU_ID = 0   # if we have multiple GPUs, pick one
#caffe.set_device(GPU_ID)
#caffe.set_mode_gpu()
net = None
cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)
correct_cls=[]
wrong_cls=[]
##############################################
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/AWA_data/Animals_with_Attributes2/discriminative_attr_class_all.txt') as inputfile:
    for row in csv.reader(inputfile):
        correct_cls.append(row[0].split(" ")[0])
        wrong_cls.append(row[0].split(" ")[1])
inputfile.closed
#Loading attributes
att_names=[]
att_index=[]
att_cls=[]
image_numb=[]
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/AWA_data/Animals_with_Attributes2/ground_clean_attr_50.txt') as inputfile:

    for row in csv.reader(inputfile):
        att_names.append (row[0].split(" ")[0])
print (len(att_names))
inputfile.closed
att_names=np.array(att_names)

#print (att_cls)
# att_cls=np.array(att_cls)


att_index=[int(x) for x in att_index]
att_index= (np.asarray(att_index))
for i in range (len(att_names)/total_no_img):
    print((att_names[i*scale:(i*scale+scale)]))

# print ((attributes))
att_names_adv=[]
att_index_adv=[]
att_cls_adv=[]
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/AWA_data/Animals_with_Attributes2/ground_adv_attr_50.txt') as inputfile:
    for row in csv.reader(inputfile):
        att_names_adv.append(row[0].split(" ")[0])

inputfile.closed
att_names_adv=np.array(att_names_adv)
# att_cls_adv=np.array(att_cls_adv)
att_index_adv=[int(x) for x in att_index_adv]
att_index_adv= (np.asarray(att_index_adv))
for i in range (len(att_names_adv)/total_no_img):
    print((att_names_adv[i*scale:(i*scale+scale)]))


def demo(image_name,image_no,image_index,net):
    colors=["blue","green","red","cyan","magenta","yellow","black","white","darkblue","orchid","springgreen","lime","deepskyblue","mediumvioletred","maroon","orangered","blue","green","red","cyan","magenta","yellow","black","white","darkblue","orchid","springgreen","lime","deepskyblue","mediumvioletred","maroon","orangered","orangered","orangered","yellow","black","white","darkblue","orchid","springgreen","orangered","blue","green","red","cyan","magenta","yellow","black","white","darkblue","orchid","springgreen","lime","deepskyblue","mediumvioletred","maroon","orangered","orangered","orangered","yellow","black","white","darkblue","orchid","springgreen","orangered","blue","green","red","cyan","magenta","yellow","black","white","darkblue","orchid","springgreen","lime","deepskyblue","mediumvioletred","maroon","orangered","orangered","orangered","yellow","black","white","darkblue","orchid","springgreen","orangered","blue","green","red","cyan","magenta","yellow","black","white","darkblue","orchid","springgreen","lime","deepskyblue","mediumvioletred","maroon","orangered","orangered","orangered","yellow","black","white","darkblue","orchid","springgreen"]
    conf_thresh=0.4
    min_boxes=36
    max_boxes=36
    indexes=[]
    cfg.TEST.NMS=0.6


    im = cv2.imread(os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/AWA_data/Animals_with_Attributes2/adv_images",image_name))
    cls_append=[]
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
    att_unique=np.unique(att_names[image_index*scale:(image_index*scale+scale)])
    att_unique_adv=np.unique(att_names_adv[image_index*scale:(image_index*scale+scale)])
    # cls_unique=np.unique(att_cls[image_index*scale:(image_index*scale+scale)])
    # cls_unique_adv=np.unique(att_cls_adv[image_index*scale:(image_index*scale+scale)])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    sizes = np.shape(im)
    height = float(sizes[0])
    width = float(sizes[1])
    fig=plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im)

    boxes = cls_boxes[keep_boxes]
    #print (boxes)
    #print (keep_boxes)
    objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    attr_thresh = 0.1
    attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)
    count_box=0
    print ("image #",image_index)
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
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("longleg","leg")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("longneck","neck")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("patches","patch")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("bulbous","round")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("lean","leaning")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("chewteeth","teeth")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("meatteeth","teeth")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("buckteeth","teeth")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("strainteeth","teeth")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("flys","flying")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("swims","swimming")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("tunnels","tunnel")
            attributes[attr[i]+1]=attributes[attr[i]+1].replace("walks","walking")

            if attributes[attr[i]+1] in att_unique_adv:
                # if (attributes[attr[i]+1]=="patch"):
                cls=attributes[attr[i]+1]
                cls=cls.replace("patch","have patches")
                    # cls = attributes[attr[i]+1]
                # elif (attributes[attr[i]+1]=="spots"):
                cls=cls.replace("spots","have spots")
                    # cls = attributes[attr[i]+1]
                # elif (attributes[attr[i]+1]=="stripes"):
                cls=cls.replace("stripes","have stripes")
                    # cls = attributes[attr[i]+1]
                # elif (attributes[attr[i]+1]=="furry"):
                cls=cls.replace("furry","have fur")
                    # cls = attributes[attr[i]+1]
                # elif (cls=="hands"):
                # elif (cls=="hands"):

                cls=cls.replace("hands","have hands")
                            # cls = cls
                        # elif (cls=="pads"):
                cls=cls.replace("pads","have pads")
                            # cls = cls
                        # elif (cls=="paws"):
                cls=cls.replace("paws","have paws")
                            # cls = cls
                        # elif (cls=="leg"):
                cls=cls.replace("leg","have leg")
                            # cls = cls
                        # elif (cls=="neck"):
                cls=cls.replace("neck","have neck")
                            # cls = cls
                        # elif (cls=="tail"):
                cls=cls.replace("tail","have tail")
                            # cls = cls
                        # elif (cls=="teeth"):
                cls=cls.replace("teeth","have teeth")
                            # cls = cls
                        # elif (cls=="horns"):
                cls=cls.replace("horns","have horns")
                            # cls = cls
                        # elif (cls=="claws"):
                cls=cls.replace("claws","have claws")
                            # cls = cls
                        # elif (cls=="tusks"):
                cls=cls.replace("tusks","have tusks")
                            # cls = cls
                        # elif (cls=="flying"):
                cls=cls.replace("flying","is flying")
                            # cls = cls
                        # elif (cls=="swimming"):
                cls=cls.replace("swimming","is swimming")
                            # cls = cls
                        # elif (cls=="tunnel"):
                cls=cls.replace("tunnel","in tunnel")
                            # cls = cls
                        # elif (cls=="walking"):
                cls=cls.replace("walking","is walking")
                            # cls = cls
                        # elif (cls=="fish"):
                cls=cls.replace("fish","eats fish")
                            # cls = cls
                        # elif (cls=="meat"):
                cls=cls.replace("meat","eats meat")
                            # cls = cls
                        # elif (cls=="desert"):
                cls=cls.replace("desert","lives in desert")
                            # cls = cls
                        # elif (cls=="bush"):
                cls=cls.replace("bush","lives in bush")
                            # cls = cls
                        # elif (cls=="plains"):
                cls=cls.replace("plains","lives in plains")
                            # cls = cls
                        # elif (cls=="forest"):
                cls=cls.replace("forest","lives in forest")
                            # cls = cls
                        # elif (cls=="fields"):
                cls=cls.replace("fields","lives in fields")
                            # cls = cls
                        # elif (cls=="mountains"):
                cls=cls.replace("mountains","lives in mountains")
                            # cls = cls
                        # elif (cls=="ocean"):
                cls=cls.replace("ocean","lives in ocean")
                            # cls = cls
                        # elif (cls=="ground"):
                cls=cls.replace("ground","lives in ground")
                            # cls = cls
                        # elif (cls=="water"):
                cls=cls.replace("water","lives in water")
                            # cls = cls
                        # elif (cls=="tree"):
                cls=cls.replace("tree","lives in tree")
                            # cls = cls
                        # elif (cls=="group"):
                cls=cls.replace("group","lives in group")
                cls=cls.replace("black","is black")
                        # cls = cls
                        # elif (str(cls)=="white"):
                cls=cls.replace("white","is white")
                        # cls = cls
                        # elif (str(cls)=="blue"):
                cls=cls.replace("blue","is blue")
                        # cls = cls
                        # elif (str(cls)=="brown"):
                cls=cls.replace("brown","is brown")
                        # cls = cls
                        # elif (str(cls)=="gray"):
                cls=cls.replace("gray","is gray")
                        # cls = cls
                        # elif (str(cls)=="orange"):
                cls=cls.replace("orange","is orange")
                        # cls = cls
                        # elif (str(cls)=="yellow"):
                cls=cls.replace("yellow","is yellow")
                        # cls = cls
                        # elif (str(cls)=="green"):
                cls=cls.replace("green","is green")
                        #     cls = cls
                        # elif (str(cls)=="red"):
                cls=cls.replace("red","is red")
                        #     cls = cls
                        # elif (cls=="furry"):
                cls=cls.replace("furry","is furry")
                cls=cls.replace("spots","have spots")
                        #     cls = cls
                        # elif (cls=="stripes"):
                cls=cls.replace("stripes","have stripes")
                cls=cls.replace("big","is big")
                cls=cls.replace("small","is small")


                # else:
                    #cls = attributes[attr[i]+1] + " " + classes[objects[i]+1]
                    # cls = attributes[attr[i]+1] + " " + correct_cls[image_index]
                #cls = attributes[attr[i]+1]
                # cls = cls.replace('brown','brown '+wrong_cls[image_no])
                # cls = cls.replace('black','black '+wrong_cls[image_no])
                # cls = cls.replace('white','white '+wrong_cls[image_no])
                # cls = cls.replace('blue','blue '+wrong_cls[image_no])
                # cls = cls.replace('gray','gray '+wrong_cls[image_no])
                # cls = cls.replace('orange','orange '+wrong_cls[image_no])
                # cls = cls.replace('red','red '+wrong_cls[image_no])
                # cls = cls.replace('yellow','yellow '+wrong_cls[image_no])
                # cls = cls.replace('patch','have patches')
                # cls = cls.replace('spots','have spots')
                cls_append.append(cls)
                    
                count = cls_append.count(cls)
                if count == 1:
                   count_box=count_box+1
                    	
                   plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor=colors[i], linewidth=0.3, alpha=0.5))
                   plt.gca().text(bbox[0], bbox[1]-2,'%s' % (cls),bbox=dict(facecolor='blue', alpha=0,linewidth=0.2),fontsize=2.5, color=colors[i])
    
        
             


    plt.savefig('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/AWA_data/Animals_with_Attributes2/adv_bb1/Adv_bb{}.jpg'.format(image_no), dpi = 1500)
    #plt.savefig('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/pytorch-adversarial_box/plots_AT_NoAT/adv_bb_AT/adv_bb_AT{}_25.jpg'.format(image_no), dpi = 1500)
    plt.close()

file_names=[sorted(glob.glob("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/AWA_data/Animals_with_Attributes2/adv_images/*.jpg"),key=lambda x:float(re.findall("([0-9]+?)\.jpg",x)[0]))]
#file_names=[sorted(glob.glob( os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/pytorch-adversarial_box/plots_AT_NoAT/adv_images",'*.jpg') ),key=lambda x:float(re.findall("([0-9]+?)\.jpg",x)[0]))]
image_name=[]
image_no=[]
image_index=[]
#import zip
for i in range (len(file_names[0])):
    image_name.append(file_names[0][i].split("/")[8])
    image_no.append(int(file_names[0][i].split("/")[8].split("Adv")[1].split(".jpg")[0]))
    image_index.append(i)
print (image_no[0:10])
print (len(image_name))
print (len(image_no))
for j in range (len(image_no)):

    demo(image_name[j],image_no[j],image_index[j],net)
