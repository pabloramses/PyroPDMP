import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import os
PATH = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()
                                ,transforms.Lambda(lambda x: torch.flatten(x, 1,2))
                                ])

train_dataset = datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)


"""PREPROCESSING"""

"""
train_data = train_dataset[0][0].squeeze()
for i in range(1, len(train_dataset)):
    cat = train_dataset[i][0].squeeze()
    train_data = torch.vstack((train_data, cat))
    if i%1000 == 0:
        print("train ", i)
"""
test_data = test_dataset[0][0].squeeze()
for i in range(1, len(test_dataset)):
    cat = test_dataset[i][0].squeeze()
    test_data = torch.vstack((test_data, cat))
    if i%1000 == 0:
        print("test ", i)

#torch.save(train_data, PATH+'/train_data.pt')
torch.save(test_data, PATH+'/test_data.pt')


train_labels = [train_dataset[0][1]]

for i in range(1, len(train_dataset)):
    train_labels.append(train_dataset[i][1])
train_labels = torch.tensor(train_labels)


test_labels = [train_dataset[0][1]]

for i in range(1, len(test_dataset)):
    test_labels.append(test_dataset[i][1])
test_labels = torch.tensor(test_labels)

torch.save(train_labels, PATH+'/train_labels.pt')
torch.save(test_labels, PATH+'/test_labels.pt')

