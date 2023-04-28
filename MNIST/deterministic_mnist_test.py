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

test_dataset = datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

"""LOAD THE TRAINED MODEL"""
state_dict = torch.load(PATH+'/models/trained_model1.pt')
model = MLP(U=400)
model.load_state_dict(state_dict)

output = test_model(model,test_loader)

correct = 0
for i in range(output.shape[0]):
    if np.argmax(output[i]) == test_dataset.__getitem__(i)[1]:
        correct +=1

print("Accuracy: ", correct/10000)
