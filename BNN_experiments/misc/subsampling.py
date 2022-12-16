import pyro
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import genpareto

x_train = -2*torch.pi*torch.rand(1000) + torch.pi
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(1000,))


class Data_set(Dataset):
    """
    A class used to instance the original Daset class from torch with specified data
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        """
        Method used to index a particular sample from the set
        """
        return self.x[index], self.y[index]

    def __len__(self):
        """
        Gives back the length of the set defined
        """
        return self.n_samples

train_set = Data_set(x_train, y_train)
train_loader = DataLoader(dataset = train_set, batch_size=10, shuffle=True)

q = 10
for i, (x,y) in enumerate(train_loader):
    if i == 0:
        x_max = torch.max(x)
    elif i == 1:
        x_max = torch.tensor((x_max,torch.max(x)))
    else:
        x_max = torch.cat((x_max, torch.max(x).expand(1)))
    if i >= q:
        break

print(x_max)
mle_estimates = genpareto.fit(x_max)
bound = genpareto.ppf(1-1/100, c = mle_estimates[0], loc = mle_estimates[1], scale = mle_estimates[2])
print(bound)