# %%
import torch
import torch.nn as nn

# %%
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)     # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)                           # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)    # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)                           # 14x14 -> 7x7

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)    # 7x7 -> 7x7

        self.flatten = nn.Flatten()                              # 64 * 7 * 7 = 3136

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)


# %%



