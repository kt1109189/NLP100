#71.単層ニューラルネットワークによる予測
import torch
from torch import nn

X_train = torch.load('./X_train.pt')
X_test = torch.load('./X_test.pt')
X_valid = torch.load('./X_valid.pt')
y_train = torch.load('./y_train.pt')
y_test = torch.load('./y_test.pt')
y_valid = torch.load('./y_valid.pt')

class SLPNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.fc(x)
        return x
    
model = SLPNet(300, 4)
y_hat_1 = torch.softmax(model(X_train[:1]), dim= -1)
print(y_hat_1)
Y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)
print(Y_hat)
