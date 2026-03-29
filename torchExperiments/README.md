# torchExperiments

PyTorch experiments comparing Euclidean and Hyperbolic neural networks on classification and regression tasks.

## Project Structure

```
main.py                                # Entry point (train + evaluate)
data.py                                # Standalone data generation script
config.py                              # All hyperparameters and dataset paths
│
├── training/                          # Training pipeline
│   ├── data_gen.py                    # Synthetic data generation (Ganea, Mircea, MNIST)
│   ├── data_loader.py                 # Data loading and model factory
│   ├── metrics.py                     # Accuracy, F1, precision, recall
│   └── trainer.py                     # Loss/optimizer factories + training loops
│
├── layers/                            # Neural network layers
│   ├── layers.py                      # Euclidean linear layer
│   ├── hyp_layers.py                  # Hyperbolic linear + activation layers
│   └── hyp_Softmax.py                 # Hyperbolic MLR (softmax equivalent)
│
├── manifolds/                         # Manifold implementations
│   ├── base.py                        # Abstract manifold + ManifoldParameter
│   ├── euclidean.py                   # Euclidean manifold
│   ├── poincare.py                    # Poincaré ball manifold
│   └── math_utils.py                  # artanh, tanh helpers
│
├── models/
│   └── hnn.py                         # HNN (Hyperbolic Neural Network)
│
├── optimizer/
│   └── Radam.py                       # Riemannian Adam
│
└── data/                              # Data files only (CSVs, no Python)
    ├── Prefix/
    └── MNIST/
```

## Installation

```bash
uv sync
```

## Usage

### Generate data

```bash
python main.py --generate_data --create_folder --task ganea --replace 0.5
```

### Train and evaluate

```bash
# Euclidean model, prefix task (dataset 10)
python main.py --train_eval --model euclidean --task ganea --loss cross --dataset 10

# Hyperbolic model, prefix task (dataset 10)
python main.py --train_eval --model hyperbolic --task ganea --loss cross --dataset 10

# Euclidean model, regression task
python main.py --train_eval --model euclidean --task mircea --loss mse --dataset 0
```

### All arguments

| Argument | Values | Description |
|---|---|---|
| `--model` | `euclidean`, `hyperbolic` | Which manifold to use |
| `--optimizer` | `Adam`, `SGD`, `Radam` | Optimizer (Radam = Riemannian Adam) |
| `--task` | `ganea`, `mircea`, `MNIST` | Task type |
| `--loss` | `cross`, `mse` | Loss function |
| `--dataset` | `0`, `10`, `30`, `50` | Prefix length (0 for mircea) |
| `--replace` | float (default 0.5) | Fraction of positive samples in ganea task |
| `--generate_data` | flag | Generate data before training |
| `--create_folder` | flag | Create output folders |
| `--train_eval` | flag | Run training and evaluation |

## TODO

- [x] Add more datasets
- [x] Implement Riemannian Adam
- [ ] Implement Riemannian SGD
- [ ] Expand to RNN

## References

- Hyperbolic Neural Networks: [paper](https://arxiv.org/abs/1805.09112) · [code](https://github.com/dalab/hyperbolic_nn)
- Poincaré Embeddings: [paper](https://papers.nips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html) · [code](https://github.com/facebookresearch/poincare-embeddings)
- Fully Hyperbolic NN: [code](https://github.com/chenweize1998/fully-hyperbolic-nn/tree/main/gcn)
- [Hyperbolic Learning Algorithms](https://github.com/drewwilimitis/hyperbolic-learning)
