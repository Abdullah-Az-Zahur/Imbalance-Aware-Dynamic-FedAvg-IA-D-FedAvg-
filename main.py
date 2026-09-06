import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.dataset import setup_ham10000_dataset, HAM10000Dataset
from src.models import get_model
from src.algorithms.ia_dfedavg import train_client_iad_fedavg, aggregate_weights
from src.utils import get_best_device, evaluate_model

def main():
    device = get_best_device()
    
    images_dir, metadata_csv = setup_ham10000_dataset()
    
    # Type Safety Check for Pylance / Pyright
    if metadata_csv is None or images_dir is None:
        raise FileNotFoundError("[ERROR] Dataset setup failed. Missing files.")
        
    df = pd.read_csv(metadata_csv)
    df['cell_type_idx'] = pd.Categorical(df['dx']).codes
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cell_type_idx'])
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = HAM10000Dataset(train_df, images_dir, transform=transform)
    test_dataset = HAM10000Dataset(test_df, images_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_clients = 5
    labels = train_df['cell_type_idx'].values.astype(np.int64)
    num_classes = len(np.unique(labels))
    
    client_indices = [[] for _ in range(num_clients)]
    dirichlet_alpha = 0.5 
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for k in range(num_classes):
        idx_k = label_indices[k]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    global_model = get_model(num_classes=7).to(device)
    rounds = 5
    history_acc, history_f1 = [], []

    print("\n--- Starting Federated Training & Evaluation ---")
    for r in range(rounds):
        print(f"\n--- Communication Round {r+1}/{rounds} ---")
        client_weights, client_sizes = [], []
        
        for client_id in range(num_clients):
            client_subset = Subset(train_dataset, client_indices[client_id])
            loader = DataLoader(client_subset, batch_size=32, shuffle=True)
            
            client_labels = labels[client_indices[client_id]]
            unique, counts = np.unique(client_labels, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            local_model = get_model(num_classes=7).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            w, loss = train_client_iad_fedavg(
                local_model, global_model, loader, class_counts, 
                epochs=1, lr=0.001, device=device
            )
            
            client_weights.append(w)
            client_sizes.append(len(client_subset))
            
        aggregated_w = aggregate_weights(client_weights, client_sizes)
        global_model.load_state_dict(aggregated_w)
        
        acc, prec, rec, f1 = evaluate_model(global_model, test_loader, device=device)
        history_acc.append(acc)
        history_f1.append(f1)
        print(f"Global Model Metrics -> Test Accuracy: {acc*100:.2f}%, F1-Score: {f1:.4f}")

    os.makedirs("./results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, rounds+1), history_acc, label='Test Accuracy', marker='o')
    plt.plot(range(1, rounds+1), history_f1, label='Test F1-Score', marker='s')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Score')
    plt.title('IA-D FedAvg Performance on HAM10000')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/performance_curve.png')
    print("\n[SUCCESS] Execution complete. Curve saved to './results/performance_curve.png'")

if __name__ == "__main__":
    main()