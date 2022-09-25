

from torch.utils.data import ConcatDataset

transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.4378, 0.4439, 0.4730), (0.1980, 0.2010, 0.1969))])

transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.4524, 0.4525, 0.4690), (0.2194, 0.2266, 0.2285))])

transform_extra = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.4300, 0.4284, 0.4427), (0.1963, 0.1979, 0.1995))])


trainset = torchvision.datasets.SVHN(root='./data', split = 'train',  transform=transform_train,
                                        download=True)

extraset = torchvision.datasets.SVHN(root='./data', split = 'extra',  transform=transform,
                                        download=True)

trainset = ConcatDataset([trainset, extraset])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)


testset = torchvision.datasets.SVHN(root='./data', split = 'test',  transform=transform_test, 
                                       download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)


