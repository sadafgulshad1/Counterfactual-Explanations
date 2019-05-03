"""
Adversarially train Resnet-152
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
import pandas as pd
import os

from collections import OrderedDict
import imageio
datapath = ''
# Hyper-parameters
param = {
    'batch_size': 16,
    'test_batch_size': 100,
    'num_epochs': 30,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 0,
}
batch_size = 16
num_workers = 4
step_size = 20
# Data loaders
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

train_dataset = CUB(root=datapath,

                    train=True,

                    transform=transform_train,

                    download=True)

test_dataset = CUB(root=datapath,

                   train=False,

                   transform=transform_test)

val_size = int(len(train_dataset) * 0.1)

train_size = len(train_dataset) - val_size

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


# Setup the model
model=torch.load("/home/sgulshad/sadaf/CUB_experiments/transfer_learn_CUB_Stephan")      
model = model.cuda()
#checkpoint = torch.load("transfer_learn_CUB_fine_S.pth")   
checkpoint = torch.load("/home/sgulshad/sadaf/CUB_experiments/transfer_learn_CUB_Stephan.pth")        
new_state_dict = OrderedDict()
#state_dict =checkpoint['state_dict']
state_dict =checkpoint
# for k, v in state_dict.items():
#     name=k
#     print (k)
#     new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)
# model.eval()

for i in range(0, len(model.state_dict().keys())):
    new_state_dict[model.state_dict().keys()[i]] = model.state_dict()[model.state_dict().keys()[i]]

model.load_state_dict(new_state_dict)

model.eval()

modules = list( model.children())[-2:]
print(modules)
net = model

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
net.train()

# Adversarial training setup
#adversary = FGSMAttack(epsilon=0.3)
adversary = LinfPGDAttack(epsilon=0.12)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=0.9, weight_decay=param['weight_decay'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

for epoch in range(param['num_epochs']):

    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

    for t, (x, y) in enumerate(train_loader):

        x_var, y_var = to_var(x), to_var(y.long())
        loss = criterion(net(x_var), y_var)

        # adversarial training
        if epoch+1 > param['delay']:
            # use predicted label to prevent label leaking
            y_pred = pred_batch(x, net)
            x_adv = adv_train(x, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var), y_var)
            loss = (loss + loss_adv) / 2

        if (t + 1) % 100 == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test(net, test_loader)

torch.save(net.state_dict(),'/home/sgulshad/sadaf/CUB_experiments/pytorch-adversarial_box/models/adv_train_PGD_ep16.pth')
torch.save(net,'/home/sgulshad/sadaf/CUB_experiments/pytorch-adversarial_box/models/adv_train_PGD_ep16')
