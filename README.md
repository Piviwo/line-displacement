# Line Displacement using Deep Learning

This repository explores two approaches for encoding and modeling line displacement.

* **Siamese LSTM Autoencoder** — a sequence-based model that predicts coordinate sequences.
* **Graph Neural Network (GNN)** — a graph neural network that predicts shift vectors for graphs.

---

## 📂 Repository Structure

```
├── data/                 
│   ├── final_dataset/                  # Contains the final training and test datasets
│   ├── preprocessing/                  # Contains Data from different preprocessing steps
│   ├── results/                        # Contains Predicted Data and Evaluation Metrics
│   ├── original 
│       └── Input Shapefiles            # Input Shapefile   
│
├── preprocessing/
│   ├── pre_seq.ipynb                   # Preprocessing sequential data
│   ├── pre_graph.ipynb                 # Preprocessing graph data
│
├── models/
│   ├── lstm_autoencoder.ipynb          # Training for LSTM Autoencoder
│   ├── siamese_lstm_autoencoder.ipynb  # Training for Siamses LSTM Autoencoder
│   ├── grpah_model.ipynb               # Training for GraphSAGE GNN models
│
├── checkpoints/                 
│   ├── autoencoder/
│   │    └── history/                   # Model history 
│   │    └── model/                     # Model files
│   │    └── weights/                   # Model weights 
│   ├── siamese/
│   │    └── history/
│   │    └── model/
│   │    └── weights/
│   ├── graph/
│        └── history/        
│        └── model
│
├── visualization/
│   ├── result_visualization.ipynb      # Calculations for Evaluation Metrics and various Visualization functions
│
└── README.md
```

---

## 📖 Citation

If you use this repository in your research, please cite it accordingly:

```bibtex
@mastersthesis{Wolffram2026Thesis,
  author       = {Wolffram, Pia},
  title        = {{Line Displacement Using Deep Learning}},
  school       = {Technical University of Munich},
  year         = {2026},
  address      = {Munich, Germany},
  month        = {February},
  type         = {Master's Thesis}, 
  url          = {https://github.com/Piviwo/line-displacement},
  note         = {Available at GitHub}
}