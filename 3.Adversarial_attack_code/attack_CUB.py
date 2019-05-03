"""Attack loop
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torchvision
import torch.utils.data as data
from scipy.misc import imsave
from dataset import Dataset, default_inception_transform
from collections import OrderedDict
from itertools import izip
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
def run_attack(args, attack):
    assert args.input_dir
    if args.targeted:
        dataset = Dataset(
            args.input_dir,
            transform=default_inception_transform(args.img_size))
    else:
        dataset = Dataset(
            args.input_dir,
            target_file='',
            transform=default_inception_transform(args.img_size))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    #model = torchvision.models.resnet18(pretrained=False)
    model=torch.load("transfer_learn_CUB_Stephan")
    if not args.no_gpu:
        model = model.cuda()
	#model = torch.nn.DataParallel(model).cuda()

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)

        new_state_dict = OrderedDict()
        #state_dict =checkpoint['state_dict']
        state_dict =checkpoint
        for k, v in state_dict.items():

            #name = k[7:] # remove module.
            #print (name)
            #print ("The k is ")
            name=k
            #print (k)
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        #if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #    model.load_state_dict(checkpoint['state_dict'])
        #else:
         #   model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % args.checkpoint_path)

    model.eval()

    for batch_idx, (input, target) in enumerate(loader):
        if not args.no_gpu:
            input = input.cuda()
            #print (target)
            target = target.cuda()

        input_adv = attack.run(model, input, target, batch_idx)
        start_index = args.batch_size * batch_idx
        indices = list(range(start_index, start_index + input.size(0)))
        ##creating the subdirectories for each class
        directory_list=[]
        directory_list1=[]
        for root, dirs, files in os.walk("/home/sgulshad/sadaf/CUB_experiments/test_split/", topdown=False):
            #print (dirs)
            for name in dirs:
                directory_list.append(( name))
                directory_list1.append(( name))
		for i in range (len(directory_list)):
			if not os.path.exists(os.path.join(args.output_dir,directory_list[i])):
				os.mkdir(os.path.join(args.output_dir,directory_list[i]))





			c=izip(dataset.filenames(indices, basename=True), input_adv)
			for filename, o in c:
				#print(filename)
				filename1=filename
				filename1=filename1.lower()
				directory_list1[i]=directory_list1[i].lower()
				if str(filename1.split('_0')[0]) == str(directory_list1[i].split('.')[1]):
					#print (filename.split('_0')[0])
					#print (o.dtype)


                                        #o1 = Image.fromarray((o*255).astype(np.uint8))
                                        #print (o.shape)

					output_file = os.path.join(args.output_dir,directory_list[i], filename)

					imsave(output_file, (o + 1.0) * 0.5, format='jpeg')
                                        #io.imsave(output_file.split(".")[0] +'.png', (o + 1.0) * 0.5)
                                        #o1.save(output_file,'JPEG')
                                        #plt.imsave(output_file,o)
