import glob, os, re
import numpy as np
import cv2, csv
import matplotlib.pyplot as plt
from PIL import Image
correct_cls=[]
wrong_cls=[]
with open('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/LAD_experiments/Analysis/Analysis_new/discriminative_attr_class_all.txt') as inputfile:
    for row in csv.reader(inputfile):
        correct_cls.append(row[0].split(" ")[0])
        wrong_cls.append(row[0].split(" ")[1])
inputfile.closed

def demo(image_name,image_no,image_name_AT,image_no_AT,image_index):
    
    im = cv2.imread(os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/LAD_experiments/Analysis/Analysis_new/clean_bb",image_name))
    im_AT = cv2.imread(os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/LAD_experiments/Analysis/Analysis_new/adv_bb",image_name_AT))
    #im = cv2.imread(os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/adv_bb",image_name))
    #im_AT = cv2.imread(os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/clean_bb",image_name_AT))
    print (im.shape)

    vis = np.concatenate((im, im_AT), axis=1)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    sizes = np.shape(vis)
    height = float(sizes[0])
    width = float(sizes[1])
    fig=plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(vis)
    plt.suptitle("Ground Truth class: "+(correct_cls[int(image_index)])+", Predicted class: "+(wrong_cls[int(image_index)]),fontsize=2)
    plt.savefig('/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/LAD_experiments/Analysis/Analysis_new/Merged/Merged{}'.format(image_no), dpi = 1500)
    plt.close()





file_names=[sorted(glob.glob( os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/LAD_experiments/Analysis/Analysis_new/clean_bb",'*.jpg') ),key=lambda x:float(re.findall("([0-9]+?)\.jpg",x)[0]))]
#file_names=[sorted(glob.glob( os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/adv_bb",'*.jpg') ),key=lambda x:float(re.findall("([0-9]+?)\.jpg",x)[0]))]
print (file_names)
image_name=[]
image_no=[]
image_index=[]
for i in range (len(file_names[0])):
    image_name.append(file_names[0][i].split("/")[9])
    image_no.append(int(file_names[0][i].split("/")[9].split("bb")[1].split(".jpg")[0]))
    image_index.append(i)
print (image_no[0:10])
#for j in range (len(image_no)):

    #demo(image_name[j],image_no[j])

##################
file_names_AT=[sorted(glob.glob( os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/LAD_experiments/Analysis/Analysis_new/adv_bb",'*.jpg') ),key=lambda x:float(re.findall("([0-9]+?)\.jpg",x)[0]))]
#file_names_AT=[sorted(glob.glob( os.path.join("/media/sadaf/e4da0f25-29be-4c9e-a432-3193ff5f5baf/Code/Pytorch_Code/transfer_learn/Analysis/clean_bb",'*.jpg') ),key=lambda x:float(re.findall("([0-9]+?)\.jpg",x)[0]))]
image_name_AT=[]
image_no_AT=[]
#import zip
for i in range (len(file_names_AT[0])):
    image_name_AT.append(file_names_AT[0][i].split("/")[9])
    image_no_AT.append(int(file_names_AT[0][i].split("/")[9].split("bb")[1].split(".jpg")[0]))
    #image_no_AT.append(i)
print (image_no_AT[0:10])
for j in range (len(image_no_AT)):

    demo(image_name[j],image_no[j], image_name_AT[j],image_no_AT[j], image_index[j])
