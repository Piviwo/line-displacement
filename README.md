# River Line Models: Sequence vs Graph

This repository explores two approaches for modeling river polylines and predicting shifts between them:

* **Siamese LSTM Autoencoder** — a sequence-based model that encodes and reconstructs river coordinate sequences.
* **Graph Attention Network (GAT)** — a graph neural network that leverages spatial and structural relationships between river points.

---

## 📂 Repository Structure

```
├── data/                 
│   ├── sequence/
│   │    └── sequences.npy           # (N, 64, 2) river coordinate sequences
│   ├── graph/
│       └── graphs_delaunay.pt       # Graphs built from sequences (PyTorch Geometric)
│       └── graphs_sequential.pt     
│   ├── original 
│       └── Input Shapefiles         # Input Shapefile   
│
├── preprocessing/
│   ├── pre_seq.ipynb                # Preprocessing sequential data
│   ├── pre_graph.ipynb              # Preprocessing graph data
│
├── models/
│   ├── siamese_lstm.py              # Training script for LSTM autoencoder
│   ├── gat_model.py                 # Training script for GAT model
│
├── checkpoints/                 
│   ├── autoencoder/
│   │    └── history/                # Model history 
│   │    └── model/                  # Model files
│   │    └── tuner_results/          # Results retrieved by Bayesian Optimization
│   │    └── weights/                # Model weights 
│   ├── siamese/
│   │    └── history/
│   │    └── model/
│   │    └── tuner_results/
│   │    └── weights/
│   ├── graph/
│        └── history/        
│        └── model
│
├── preprocessing/
│   ├── data_visualization.ipynb     # Jupyter Notebook for visualizing the input data 
│   ├── result_visualization.ipynb   # Jupyter Notebook for visualizing the results
│
├── environment.yml                  # Conda Environment Description 
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

* **Nodes**: points along vector line
* **Node features**: line ID, `(x, y)` coordinates, sequence length, distances to other lines, angle
* **Edges**:

  * Sequential (chain)
  * Hybrid: Sequential & Delaunay triangulation

---

## 🚀 Usage

### Preprocessing

<!-- ```bash
# For sequence data
python preprocessing/pre_seq.py

# For graph data
python preprocessing/pre_graph.py
``` -->

### Training

<!-- ```bash
# Train Siamese LSTM Autoencoder
python train_lstm.py

# Train Graph Attention Network
python train_gat.py
```

--- -->

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
