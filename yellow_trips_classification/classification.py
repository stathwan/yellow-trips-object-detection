
from read_dataset import import_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchsummary import summary  # install pip install torchsummary


epochs=10
batch_size=64
test_batch_size=1000
lr=0.01
momentum=0.5
seed=1
save_interval=10

no_cuda=True
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



class customdataset



class Convolution_net(nn.Module):
    def __init__(self):
        super(Convolution_net, self).__init__()
        self.conv1_1 = nn.Conv2d(1 , 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3)
        
        self.conv4_1 = nn.Conv2d(128, 10, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(F.relu(self.conv1_2(x)),2)

        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(F.relu(self.conv2_2(x)),2)
        x = F.dropout(x,training=0.5)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))

        x = self.conv4_1(x)
        x = x.view(-1,10)        
        
        return F.log_softmax(x, dim=0)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % save_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item() # sum up batch loss ,reduction='sum', item() pytorch to python object
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

   
    

### load data
path='./own_dataset'
train_x, train_y, val_x, val_y =import_data(path=path, shuffle=True, one_hot_encoding=True, split=True)


### build&compile model
model = Convolution_net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

### fit model
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
