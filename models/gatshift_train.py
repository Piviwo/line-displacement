import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader


# ================================================================
# Model Definition
# ================================================================

class GATShift(nn.Module):
    """
    GATv2-based node regression: predicts (dx, dy) per node.
    Uses edge_attr through GATv2Conv(edge_dim=...).
    """

    def __init__(self, in_dim, edge_dim, hidden=64, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activ = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Input layer
        self.layers.append(
            GATv2Conv(in_channels=in_dim, out_channels=hidden, heads=heads,
                      edge_dim=edge_dim, dropout=dropout, concat=True)
        )
        out_dim = hidden * heads

        # Hidden layers
        for _ in range(layers - 2):
            self.layers.append(
                GATv2Conv(in_channels=out_dim, out_channels=hidden, heads=heads,
                          edge_dim=edge_dim, dropout=dropout, concat=True)
            )
            out_dim = hidden * heads

        # Output GAT layer (concat=False)
        self.layers.append(
            GATv2Conv(in_channels=out_dim, out_channels=hidden, heads=1,
                      edge_dim=edge_dim, dropout=dropout, concat=False)
        )

        # Regression head -> (dx, dy)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, gat in enumerate(self.layers):
            x = gat(x, edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:
                x = self.activ(x)
                x = self.dropout(x)

        return self.head(x)  # [N, 2]


# ================================================================
# Training & Evaluation
# ================================================================

def train_epoch(model, loader, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)

        # Mask: only synthetic_1 nodes (line_id == 1)
        line_ids = batch.x[:, 0]
        mask = (line_ids == 1)

        if mask.sum() == 0:
            continue  # skip if no nodes to supervise

        loss = F.mse_loss(pred[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def eval_epoch(model, loader, device):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            line_ids = batch.x[:, 0]
            mask = (line_ids == 1)

            if mask.sum() == 0:
                continue

            loss = F.mse_loss(pred[mask], batch.y[mask])
            total_loss += loss.item()

    return total_loss / max(1, len(loader))


# ================================================================
# Inference Utilities
# ================================================================

def infer_graph(model, graph, device):
    """
    Run inference on a single graph.
    Returns predicted shifts and corrected coordinates for synthetic_1 nodes.
    """
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        pred_shift = model(graph).cpu().numpy()  # [N, 2]

        line_ids = graph.x[:, 0].cpu().numpy()
        mask = line_ids == 1
        coords = graph.x[:, 1:3].cpu().numpy()

        corrected = coords.copy()
        corrected[mask] += pred_shift[mask]

    return pred_shift, corrected, mask


# ================================================================
# Utility Functions
# ================================================================

def load_graphs(path):
    """Load graphs safely from a .pt file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")

    graphs = torch.load(path, weights_only=False)
    if not isinstance(graphs, list):
        graphs = [graphs]

    if len(graphs) == 0:
        raise ValueError("No graphs found in the dataset.")

    print(f"✅ Loaded {len(graphs)} graphs.")
    return graphs


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pt"):
    """Save model and optimizer state."""
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)
    print(f"💾 Saved checkpoint: {path}")


# ================================================================
# Main Function
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GATShift for node-level regression.")
    parser.add_argument("--data", type=str, default="../data/final_dataset/graph/graphs_sequential.pt",
                        help="Path to graph dataset (.pt).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--layers", type=int, default=3, help="Number of GAT layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs.")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load data
    graphs = load_graphs(args.data)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    # Initialize model and optimizer
    sample = graphs[0]
    model = GATShift(
        in_dim=sample.x.size(1),
        edge_dim=sample.edge_attr.size(1) if sample.edge_attr is not None else 0,
        hidden=args.hidden, heads=args.heads, layers=args.layers, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    print("🧠 Starting training...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        val_loss = eval_epoch(model, loader, device)  # TODO: use a separate validation set

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:03d}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, f"checkpoint_epoch{epoch}.pt")

    # Inference on first graph
    pred_shift, corrected, mask = infer_graph(model, graphs[0], device)
    print("\n🔍 Inference on first graph:")
    print("Predicted first 3 shifts (synthetic_1 nodes):", pred_shift[mask][:3])
    print("Corrected first 3 coordinates:", corrected[mask][:3])


if __name__ == "__main__":
    main()