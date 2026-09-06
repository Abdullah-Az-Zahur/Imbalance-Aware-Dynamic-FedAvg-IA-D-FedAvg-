import copy
import torch
import torch.nn as nn

def train_client_fedavg(model, train_loader, epochs=1, lr=0.001, device='cpu'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss = torch.tensor(0.0, device=device)
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
    return model.state_dict(), loss.item()