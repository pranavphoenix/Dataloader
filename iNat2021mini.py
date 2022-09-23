import torch, torchvision
import torchvision.transforms as transforms

!wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
!wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
  
!tar -xzvf "/content/train_mini.tar.gz" -C "/content/" 
!tar -xzvf "/content/val.tar.gz" -C "/content/" 

batch_size = 96

transform_train = transforms.Compose(
        [ transforms.Resize([256,256]),
            transforms.ToTensor().
         transforms.Normalize((0.4642, 0.4807, 0.3768), (0.2345, 0.2255, 0.2442)) ])

transform_test = transforms.Compose(
        [ transforms.Resize([256,256]),
            transforms.ToTensor().
         transforms.Normalize((0.4643, 0.4808, 0.3769), (0.2349, 0.2259, 0.2445)) ])

trainset = torchvision.datasets.ImageFolder(root='/content/train_mini', transform=transform_train)
testset = torchvision.datasets.ImageFolder(root='/content/val', transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
