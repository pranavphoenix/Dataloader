#loads images as 3*64*64 tensors 
from PIL import Image

!wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
!unzip -q tiny-imagenet-200.zip

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

#change /workspace/ to current folder

id_dict = {}
for i, line in enumerate(open('/workspace/tiny-imagenet-200/wnids.txt', 'r')):
  id_dict[line.replace('\n', '')] = i

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/workspace/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)

        if image.mode == "L":
          image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image)
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/workspace/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('/workspace/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == "L":
          image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label


transform_train = transforms.Compose(
        [ 
            transforms.ToTensor(),
     transforms.Normalize((0.4803, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))])

transform_test = transforms.Compose(
        [ 
            transforms.ToTensor(),
     transforms.Normalize((0.4823, 0.4495, 0.3982), (0.2771, 0.2694, 0.2830))])

batch_size = 64

trainset = TrainTinyImageNetDataset(id=id_dict, transform = transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = TestTinyImageNetDataset(id=id_dict, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
