# River Line Models: Sequence vs Graph

This repository explores two approaches for modeling river polylines and predicting shifts between them:

* **Siamese LSTM Autoencoder** — a sequence-based model that encodes and reconstructs river coordinate sequences.
* **Graph Attention Network (GAT)** — a graph neural network that leverages spatial and structural relationships between river points.

---

## 📂 Repository Structure

```
├── data/                 
│   ├── sequences.npy      # (N, 64, 2) river coordinate sequences
│   └── graphs.pt          # Graphs built from sequences (PyTorch Geometric)
├── preprocessing/
│   ├── pre_seq.py         # Preprocessing sequential data
│   ├── pre_graph.py       # Preprocessing graph data
├── models/
│   ├── siamese_lstm.py    # Siamese LSTM Autoencoder
│   ├── gat_model.py       # Graph Attention Network implementation
├── train_lstm.py          # Training script for LSTM autoencoder
├── train_gat.py           # Training script for GAT model
├── utils.py               # Preprocessing, graph construction, visualization
└── README.md
```

---

## 🗂 Dataset

### Sequences

Stored in `sequences.npy` as numpy arrays of shape **(N, 64, 2)**:

* **N**: number of river sequences
* **64**: number of sampled points per polyline
* **2**: coordinates `(x, y)`

### Graphs

Stored in `graphs.pt` as PyTorch Geometric objects.

* **Nodes**: river points
* **Node features**: line ID, `(x, y)` coordinates, sequence length, distances to other lines
* **Edges**:

  * Sequential (chain)
  * Delaunay triangulation
  * Hybrid

---

## 🚀 Usage

### Preprocessing

```bash
# For sequence data
python preprocessing/pre_seq.py

# For graph data
python preprocessing/pre_graph.py
```

### Training

```bash
# Train Siamese LSTM Autoencoder
python train_lstm.py

# Train Graph Attention Network
python train_gat.py
```

---

## 🛠 Requirements

* Python 3.8+
* PyTorch
* PyTorch Geometric
* NumPy
* SciPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📖 Citation

If you use this repository in your research, please cite it accordingly (bibtex entry to be added).

---

## 📜 License
