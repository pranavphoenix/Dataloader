import torchvision, torch
import torchvision.transforms as transforms


transform_train = transforms.Compose(
        [
        transforms.ToTensor(),
     	transforms.Normalize((0.2862), (0.3532))])

transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
     	transforms.Normalize((0.2869), (0.3525))])



batch_size = 1024
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, 
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
