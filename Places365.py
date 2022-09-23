import torch

import torchvision
import torchvision.transforms as transforms


!wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
!tar -xvf "/content/places365standard_easyformat.tar" -C "/content/"     #[run this cell to extract tar files]

batch_size = 96

transform_test = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.ToTensor(),
     transforms.Normalize((0.4576, 0.4411, 0.4080), (0.2689, 0.2669, 0.2849))])

transform_train = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.ToTensor(),
     transforms.Normalize((0.4577, 0.4413, 0.4078), (0.2695, 0.2671, 0.2853))])

trainset = torchvision.datasets.ImageFolder(root='/content/places365_standard/train', transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = torchvision.datasets.ImageFolder(root='/content/places365_standard/val', transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
