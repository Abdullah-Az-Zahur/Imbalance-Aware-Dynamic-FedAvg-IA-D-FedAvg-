import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    else:
        print("[INFO] No compatible GPU found. Falling back to CPU.")
        return torch.device("cpu")

def calculate_dynamic_mu(class_counts: dict, base_mu: float = 0.01) -> float:
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        return base_mu
    
    probs = [count / total_samples for count in class_counts.values() if count > 0]
    entropy = -sum(p * np.log(p + 1e-8) for p in probs)
    max_entropy = np.log(len(class_counts)) if len(class_counts) > 0 else 1.0
    
    imbalance_factor = 1.0 - (entropy / (max_entropy + 1e-8))
    return base_mu * (1.0 + imbalance_factor)

class ImbalanceAwareLoss(nn.Module):
    def __init__(self, class_counts: list, device: torch.device | str = 'cpu'):
        super(ImbalanceAwareLoss, self).__init__()
        total_samples = sum(class_counts)
        class_weights = [total_samples / (c + 1e-5) for c in class_counts]
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)

def evaluate_model(model: nn.Module, test_loader, device: torch.device | str = 'cpu'):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    return acc, precision, recall, f1