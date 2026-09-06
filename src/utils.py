import torch
import torch.nn as nn
import numpy as np

def calculate_dynamic_mu(class_counts, base_mu=0.01):
    """
    Calculates client-specific dynamic regularization parameter (mu) 
    based on class imbalance entropy.
    """
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        return base_mu
    
    # Calculate Class Distribution Probabilities
    probs = [count / total_samples for count in class_counts.values() if count > 0]
    
    # Imbalance Score via Shannon Entropy
    entropy = -sum(p * np.log(p + 1e-8) for p in probs)
    max_entropy = np.log(len(class_counts))
    
    # Higher Imbalance -> Lower Normalized Entropy -> Higher Penalty (Mu)
    imbalance_factor = 1.0 - (entropy / (max_entropy + 1e-8))
    dynamic_mu = base_mu * (1.0 + imbalance_factor)
    
    return dynamic_mu

class ImbalanceAwareLoss(nn.Module):
    def __init__(self, class_counts, device='cpu'):
        super(ImbalanceAwareLoss, self).__init__()
        total_samples = sum(class_counts)
        # Class Weights for Cross Entropy (Inverse Frequency)
        class_weights = [total_samples / (c + 1e-5) for c in class_counts]
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        # Normalize weights
        weights_tensor = weights_tensor / weights_tensor.sum()
        
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)