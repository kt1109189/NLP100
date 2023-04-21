#72.損失と勾配の計算
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

criterion = nn.CrossEntropyLoss()
l_1 = criterion(model(X_train[:1]), y_train[:1])
model.zero_grad()
l_1.backward()

print(f'損失: {l_1:.4f}')
print(f'勾配:\n{model.fc.weight.grad}')

l = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()

print(f'損失: {l:.4f}')
print(f'勾配:\n{model.fc.weight.grad}')

