import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

torch.manual_seed(0)
x_train = -2*torch.pi*torch.rand(100) + torch.pi
torch.manual_seed(0)
y_train = torch.sin(x_train) + torch.normal(0., 0.1, size=(100,))

torch.manual_seed(1)
x_test = -2*torch.pi*torch.rand(100) + torch.pi
torch.manual_seed(1)
y_test = torch.sin(x_test)

class dnn(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.hidden = nn.Linear(1, L)
        self.out = nn.Linear(L,1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.out(x)





criterion = nn.MSELoss()

def train(train_steps, model, optimiser):
    model.train()
    for i in range(train_steps):
        output = model(x_train[0].unsqueeze(0))
        for n in range(1,100):
            output = torch.cat((output, model(x_train[n].unsqueeze(0))))

        optimiser.zero_grad()
        loss = criterion(output, y_train)
        loss.backward()
        optimiser.step()

        if i % 10 == 0:
            print("Iteration ",i, " current loss ", loss.item())

def test(model):
    model.eval()

    output = model(x_test[0].unsqueeze(0))
    for n in range(1,100):
        output = torch.cat((output, model(x_test[n].unsqueeze(0))))
    return criterion(output, y_test), output



det_nn_10 = dnn(10)
det_nn_50 = dnn(50)
det_nn_100 = dnn(100)
optimiser_10 = Adam(det_nn_10.parameters(), lr=0.005)
optimiser_50 = Adam(det_nn_50.parameters(), lr=0.005)
optimiser_100 = Adam(det_nn_100.parameters(), lr=0.005)
train(1000, det_nn_10, optimiser_10)
rsq_10, out = test(det_nn_10)
train(1000, det_nn_50, optimiser_50)
rsq_50, out = test(det_nn_50)
train(1000, det_nn_100, optimiser_100)
rsq_100, out = test(det_nn_100)

print("losses were ", rsq_10,", ",rsq_50,", ",rsq_100)
