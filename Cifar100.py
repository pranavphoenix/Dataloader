import torchvision, torch
import torchvision.transforms as transforms
batch_size = 512

transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2674, 0.2564, 0.2762))])

transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.5086, 0.4874, 0.4418), (0.2682, 0.2572, 0.2771))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
