import torchvision
import torchvision.transforms as transforms
import torch

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.ToPILImage(),  
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

trainset = torchvision.datasets.MNIST(root = "data/mnist/", train= True, download = True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root = "data/mnist/", train= False, download = True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)