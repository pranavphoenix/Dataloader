import torchvision, torch
import torchvision.transforms as transforms
batch_size = 512

transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.4466, 0.4397, 0.4066), (0.2605, 0.2567, 0.2713))])

transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.4470, 0.4395, 0.4048), (0.2605, 0.2566, 0.2699))])

trainset = torchvision.datasets.STL10(root='./data', split = 'train',  transform=transform_train,
                                        download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = torchvision.datasets.STL10(root='./data', split = 'test',  transform=transform_test, 
                                       download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
