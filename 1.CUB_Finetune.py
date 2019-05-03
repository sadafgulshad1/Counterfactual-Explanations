import os
import pandas as pd
import imageio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#cpu_device = torch.device('cpu')
from torch.autograd import Variable
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

model = models.resnet152(pretrained=True)

cnn_output_size = model.fc.in_features

model.fc = nn.Linear(cnn_output_size, num_classes)

if use_gpu:
    model = model.cuda()

print(model)

if not train:

    model.load_state_dict(torch.load(eval_model))



# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
params = filter(lambda p: p.requires_grad, model.parameters())
#optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
def run_model(model, images, labels, train=True):
    classification = model(images)
    if train:
        loss = criterion(classification, labels)
        return loss
    return classification

def test_model(model, data_loader, prefix='test'):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    
    for images, labels in data_loader:
    	images=Variable(images.cuda())
        labels=Variable(labels.cuda())
        classification = run_model(model, images, labels, train=False)
        _, predicted = torch.max(classification.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('{} accuracy of the model on the {} {} images: {:.2f}%'.format(prefix, len(data_loader)*batch_size, prefix, 100 * correct / total))
    model.train()  # Change model to 'train' mode
    return correct / total

def train_model(model, data_loader):
    max_accuracy = 0
    # Train the Model
    for epoch in range(num_epochs):
        scheduler.step()
        for i, (images, labels) in enumerate(data_loader):
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            optimizer.zero_grad()
            loss = run_model(model, images, labels, train=True)
            loss.backward()
            optimizer.step()
            if (i+1) % out_freq == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                       %(epoch+1, num_epochs, i+1,
                           len(train_dataset)//batch_size, loss.data[0]))

        val_accuracy = test_model(model, val_loader, 'validation')
        # Uncomment these lines if you want to calculate train/test set accuracy after each iteration (slows down training)
        #train_accuracy = test_model(model, train_loader, 'train')
        #test_accuracy = test_model(model, test_loader, 'test')
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            torch.save(model.state_dict(), '{}.pth'.format(name))
            
            print('Saved best model')
    torch.save(model.state_dict(),'transfer_learn_CUB_Stephan.pth')
    torch.save(model,'transfer_learn_CUB_Stephan')
    test_model(model, test_loader)
if train:
    train_model(model, train_loader)
else:
    test_model(model, test_loader)
