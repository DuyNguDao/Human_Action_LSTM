import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, num_classes, device='cpu'):
        super(RNN, self).__init__()
        if device == 'cpu':
            self.device = device
        else:
            self.device = "cuda:0"
        self.hidden_size = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device).float()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device).float()
        out, _ = self.lstm(x.float(), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    model = RNN(num_classes=2, input_size=34)
    x = torch.rand(30, 34)
    x1 = torch.rand(30, 34)
    class_name = ['Standing', 'Stand up', 'Sitting', 'Sit down', 'Lying Down', 'Walking', 'Fall Down']
    # predict with batch_size
    x = torch.stack([x, x1])
    print(x.shape)
    y = model(x)
    a, preds = torch.max(y, 1)
    print(preds.flatten().tolist())
    print(class_name[preds.flatten().tolist()])
    percentage = (nn.functional.softmax(y, dim=1)[0] * 100).tolist()
    print(preds)

