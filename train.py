from models import BaselineModel
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAUROC


def train(model, train_loader, val_loader, device, num_epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()  

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            position=0,
            leave=True,
        ) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        avg_loss= evaluate(model, val_loader, criterion, device)
        print(
            f"Validation set: Average loss = {avg_loss:.4f}"
        )


def evaluate(model, test_loader, criterion, device):
    model.eval() 

    with torch.no_grad():
        total_loss = 0.0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    return avg_loss


def test_auroc(model, test_loader, device, num_classes=5):
    model.eval()  
    auroc_metric = MultilabelAUROC(num_labels=num_classes)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            auroc_metric.update(outputs, labels)

    auroc = auroc_metric.compute()
    return auroc.item()


model = BaselineModel()

from torch.utils.data import DataLoader, TensorDataset, random_split

num_samples = 1000
num_classes = 5 
train_ratio = 0.7  
val_ratio = 0.20
test_ratio = 0.10  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.randn(num_samples, 3, 224, 224)  
labels = torch.randint(
    0, 2, (num_samples, num_classes)
)  
dataset = TensorDataset(inputs, labels)

train_size = int(train_ratio * num_samples)
val_size = int(val_ratio * num_samples)
test_size = num_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train(model=model, train_loader=train_loader, val_loader=val_loader, device=device, num_epochs=2)
print(test_auroc(model=model, test_loader=test_loader, device=device))
