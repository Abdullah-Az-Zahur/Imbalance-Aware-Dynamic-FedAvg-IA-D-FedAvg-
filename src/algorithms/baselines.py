import copy
import torch
import torch.nn as nn

def train_client_fedavg(model, train_loader, epochs=1, lr=0.001, device='cpu'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    total_loss_val = 0.0
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss_val += loss.item()
            
    return model.state_dict(), total_loss_val / len(train_loader)

def train_client_fedprox(model, global_model, train_loader, mu=0.01, epochs=1, lr=0.001, device='cpu'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    total_loss_val = 0.0
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            task_loss = criterion(outputs, targets)
            
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2) ** 2
            
            total_loss = task_loss + (mu / 2.0) * proximal_term
            total_loss.backward()
            optimizer.step()
            total_loss_val += total_loss.item()
            
    return model.state_dict(), total_loss_val / len(train_loader)