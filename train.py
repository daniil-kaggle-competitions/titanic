from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from TitanicDataset import TitanicDataset
import torch.nn as nn
import torch.nn.functional as F
import torch

class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return F.sigmoid(self.stack(x))

def epoch(train_loader, model, loss_fn, optimizer):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        

def train(train_loader, validation_loader, n_epochs):
    loss_fn = nn.BCELoss()
    model = TitanicModel()
    optimizer = Adam(model.parameters(), lr=0.003)
    min_loss = 1000000

    for e in range(n_epochs):
        print(f"Epoch {e + 1}")
        model.train(True)

        epoch(train_loader, model, loss_fn, optimizer)

        model.eval()

        loss = 0
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss += loss_fn(outputs, labels)

        avg_loss = loss / (i + 1)
        print(f"Validation Loss: {loss / (i + 1)}")
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), "best.pt")


def main():
    train_dataset, validation_dataset = random_split(TitanicDataset("train.csv"), [0.85, 0.15])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    train(train_loader, validation_loader, 50)
    

if __name__ == "__main__":
    main()