import copy
import torch
from src.utils import ImbalanceAwareLoss, calculate_dynamic_mu

def train_client_iad_fedavg(model, global_model, train_loader, class_counts, epochs=1, lr=0.001, base_mu=0.01, device='cpu'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 1. Dynamic Mu calculation based on client imbalance
    dynamic_mu = calculate_dynamic_mu(class_counts, base_mu=base_mu)
    
    # 2. Imbalance-Aware Loss Function
    counts_list = [class_counts.get(i, 0) for i in range(7)]
    criterion = ImbalanceAwareLoss(counts_list, device=device)
    total_loss = torch.tensor(0.0, device=device)
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(inputs)
            task_loss = criterion(outputs, targets)
            
            # Dynamic Proximal Regularization Term
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2) ** 2
            
            # Total Loss Calculation
            total_loss = task_loss + (dynamic_mu / 2.0) * proximal_term
            
            total_loss.backward()
            optimizer.step()
            
    return model.state_dict(), total_loss.item()


def aggregate_weights(client_weights, client_sizes):
    """ Weighted average based on client dataset sizes """
    total_size = sum(client_sizes)
    global_weights = copy.deepcopy(client_weights[0])
    
    for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
        for i in range(len(client_weights)):
            weight_factor = client_sizes[i] / total_size
            global_weights[key] += client_weights[i][key].float() * weight_factor
            
    return global_weights