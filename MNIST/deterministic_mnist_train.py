import torch
from torchvision import datasets, transforms
from torchsummary import summary
from utils import *
import os
PATH = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""LOAD MNIST DATA WITH NO PRE-PROCESSING"""
transform = transforms.Compose([transforms.ToTensor()
                                ,transforms.Lambda(lambda x: torch.flatten(x, 1,2))
                                ])

train_dataset = datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)


print(train_dataset.__getitem__(0)[0].shape)
"""CREATE THE MODEL"""
class MLP(torch.nn.Module):

    def __init__(self, U):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(28*28, U)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(U, U)
        self.linear3 = torch.nn.Linear(U, 10)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

mlp = MLP(U=400)
summary(mlp,(1,28*28))

#Hyperparametres
num_epochs = 300
learning_rate = 0.01
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr = learning_rate, weight_decay=0.01)
model, train_tracker, test_tracker = train_model(model=mlp, optimizer=optimizer,criterion=criterion, train_loader=train_loader, val_loader=test_loader, num_epochs=num_epochs)

np.save(PATH+'/models/train_track.npy',train_tracker)
np.save(PATH+'/models/test_track.npy',test_tracker)
torch.save(model.state_dict(), PATH+'/models/trained_model1.pt')