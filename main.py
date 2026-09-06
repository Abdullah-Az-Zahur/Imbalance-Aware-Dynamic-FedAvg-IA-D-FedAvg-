import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.dataset import setup_ham10000_dataset, HAM10000Dataset
from src.models import get_model
from src.algorithms.ia_dfedavg import train_client_iad_fedavg, aggregate_weights
from src.algorithms.fedavg import train_client_fedavg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    # Step 1: Setup Dataset
    images_dir, metadata_csv = setup_ham10000_dataset()
    if metadata_csv is None:
        raise FileNotFoundError("HAM10000 metadata CSV path was not provided.")
    df = pd.read_csv(metadata_csv)

    # Encode labels
    df["cell_type_idx"] = pd.Categorical(df["dx"]).codes
    labels = df["cell_type_idx"].values.astype(np.int64)

    # Image Transformations
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = HAM10000Dataset(df, images_dir, transform=transform)

    # Step 2: Create Non-IID Dirichlet Data Split for 5 Clients
    num_clients = 5
    labels = df["cell_type_idx"].to_numpy(dtype=np.int64)
    num_classes = int(df["cell_type_idx"].nunique())

    client_indices = [[] for _ in range(num_clients)]
    # Dirichlet distribution for non-IID splitting
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

    print("[INFO] Non-IID Dataset Splitting Complete.")

    # Step 3: Run Baseline or IA-D FedAvg Simulation
    print("\n--- Starting Training Process ---")
    global_model = get_model(num_classes=7).to(device)
    rounds = 3  # Initial Test Rounds

    for r in range(rounds):
        print(f"\n--- Communication Round {r+1}/{rounds} ---")
        client_weights = []
        client_sizes = []

        for client_id in range(num_clients):
            client_subset = Subset(dataset, client_indices[client_id])
            loader = DataLoader(client_subset, batch_size=32, shuffle=True)

            # Count classes for dynamic mu calculation
            client_labels = labels[client_indices[client_id]]
            unique, counts = np.unique(client_labels, return_counts=True)
            class_counts = dict(zip(unique, counts))

            # Local Model Training using Proposed IA-D FedAvg
            local_model = get_model(num_classes=7).to(device)
            local_model.load_state_dict(global_model.state_dict())

            w, loss = train_client_iad_fedavg(
                local_model,
                global_model,
                loader,
                class_counts,
                epochs=1,
                lr=0.001,
                device=str(device),
            )

            client_weights.append(w)
            client_sizes.append(len(client_subset))
            print(f"Client {client_id+1} Training Loss: {loss:.4f}")

        # Global Aggregation
        aggregated_w = aggregate_weights(client_weights, client_sizes)
        global_model.load_state_dict(aggregated_w)
        print(f"Round {r+1} Aggregation Complete.")


if __name__ == "__main__":
    main()
