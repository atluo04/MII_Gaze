from models import BaselineModel
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, train_loader, val_loader, device, num_epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(lr=0.001)
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
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                outputs = (outputs >= 0.5).float()
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
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = (outputs > 0.5).float()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    return avg_loss

model = BaselineModel()

