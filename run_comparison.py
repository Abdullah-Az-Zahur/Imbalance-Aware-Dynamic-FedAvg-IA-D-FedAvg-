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
from src.algorithms.baselines import train_client_fedavg, train_client_fedprox
from src.utils import get_best_device, evaluate_model

def run_experiment(algo_name, rounds, num_clients, train_dataset, test_loader, client_indices, labels, device):
    print(f"\n=======================================================")
    print(f"      Running Algorithm: {algo_name}")
    print(f"=======================================================")
    
    global_model = get_model(num_classes=7).to(device)
    history_acc, history_f1 = [], []

    for r in range(rounds):
        client_weights, client_sizes = [], []
        
        for client_id in range(num_clients):
            client_subset = Subset(train_dataset, client_indices[client_id])
            loader = DataLoader(client_subset, batch_size=32, shuffle=True)
            
            local_model = get_model(num_classes=7).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            if algo_name == 'FedAvg':
                w, loss = train_client_fedavg(local_model, loader, epochs=1, lr=0.001, device=device)
            elif algo_name == 'FedProx':
                w, loss = train_client_fedprox(local_model, global_model, loader, mu=0.01, epochs=1, lr=0.001, device=device)
            elif algo_name == 'IA-D FedAvg (Ours)':
                client_labels = labels[client_indices[client_id]]
                unique, counts = np.unique(client_labels, return_counts=True)
                class_counts = dict(zip(unique, counts))
                w, loss = train_client_iad_fedavg(local_model, global_model, loader, class_counts, epochs=1, lr=0.001, device=device)
            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")

            client_weights.append(w)
            client_sizes.append(len(client_subset))
            
        aggregated_w = aggregate_weights(client_weights, client_sizes)
        global_model.load_state_dict(aggregated_w)
        
        acc, prec, rec, f1 = evaluate_model(global_model, test_loader, device=device)
        history_acc.append(acc * 100)
        history_f1.append(f1 * 100)
        
        print(f"Round {r+1:02d}/{rounds} | {algo_name} -> Test Acc: {acc*100:.2f}%, F1-Score: {f1*100:.2f}%")

    return history_acc, history_f1

def main():
    device = get_best_device()
    images_dir, metadata_csv = setup_ham10000_dataset()
    
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
    
    # Fix seed for reproducible non-IID split across all algorithms
    np.random.seed(42)
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

    rounds = 50

    # Run Benchmarks
    acc_fedavg, f1_fedavg = run_experiment('FedAvg', rounds, num_clients, train_dataset, test_loader, client_indices, labels, device)
    acc_fedprox, f1_fedprox = run_experiment('FedProx', rounds, num_clients, train_dataset, test_loader, client_indices, labels, device)
    acc_iad, f1_iad = run_experiment('IA-D FedAvg (Ours)', rounds, num_clients, train_dataset, test_loader, client_indices, labels, device)

    # Save Comparison Plot
    os.makedirs("./results", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, rounds + 1), acc_fedavg, label='Standard FedAvg', linestyle='--', color='#7f7f7f', linewidth=2)
    plt.plot(range(1, rounds + 1), acc_fedprox, label='FedProx (\u03bc=0.01)', linestyle='-.', color='#ff7f0e', linewidth=2)
    plt.plot(range(1, rounds + 1), acc_iad, label='IA-D FedAvg (Ours)', linestyle='-', color='#2ca02c', linewidth=2.5)
    
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Performance Comparison under Non-IID & Class Imbalance (HAM10000)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('./results/comparison_accuracy.png', dpi=300)
    plt.close()

    # Save Comparison DataFrame
    df_comp = pd.DataFrame({
        'Round': range(1, rounds + 1),
        'FedAvg_Acc': acc_fedavg,
        'FedProx_Acc': acc_fedprox,
        'IA_D_FedAvg_Acc': acc_iad,
        'FedAvg_F1': f1_fedavg,
        'FedProx_F1': f1_fedprox,
        'IA_D_FedAvg_F1': f1_iad
    })
    df_comp.to_csv('./results/baseline_comparison_results.csv', index=False)
    print("\n[SUCCESS] Baseline Comparison Completed! Results and graph saved to './results/comparison_accuracy.png'")

if __name__ == "__main__":
    main()