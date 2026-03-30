# torchExperiments

PyTorch experiments comparing Euclidean and Hyperbolic neural networks on classification and regression tasks.

## Project Structure

```
main.py                                # Entry point (train + evaluate)
data.py                                # Standalone data generation script
config.py                              # All hyperparameters and dataset paths
experiment_config.py                   # YAML config loading + sweep expansion
│
├── experiments/                       # YAML experiment configs
│   ├── ganea_sweep.yaml               # Grid sweep for prefix task
│   ├── mircea_single.yaml             # Single regression experiment
│   └── mnist_architectures.yaml       # Architecture sweep for MNIST
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
│   └── hyp_softmax.py                 # Hyperbolic MLR (softmax equivalent)
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
├── tests/                             # Test suite (pytest)
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
uv run python data.py --task ganea --replace 0.5 --create_folder
uv run python data.py --task mircea
uv run python data.py --task MNIST
```

### Train and evaluate

```bash
# Euclidean model, prefix task (dataset 10), 5 runs
uv run python main.py --model euclidean --task ganea --loss cross --dataset 10 --runs 5

# Hyperbolic model, prefix task (dataset 10), Riemannian Adam
uv run python main.py --model hyperbolic --task ganea --loss cross --dataset 10 --optimizer Radam

# Regression task (Mircea phylogenetic)
uv run python main.py --model hyperbolic --task mircea --loss mse --optimizer Radam

# MNIST classification
uv run python main.py --model hyperbolic --task MNIST --loss cross --optimizer Radam

# With autograd anomaly detection (slow, for debugging)
uv run python main.py --model hyperbolic --task ganea --loss cross --dataset 10 --debug
```

### YAML experiment configs

```bash
# Single experiment from YAML
uv run python main.py --config experiments/mircea_single.yaml

# Full grid sweep (12 experiments × 3 runs = 36 runs)
uv run python main.py --config experiments/ganea_sweep.yaml

# Preview sweep without running
uv run python main.py --config experiments/ganea_sweep.yaml --dry-run

# Filter to specific combinations
uv run python main.py --config experiments/ganea_sweep.yaml --filter model=hyperbolic,optimizer=Radam

# Run single experiment by index
uv run python main.py --config experiments/ganea_sweep.yaml --experiment 0
```

See `experiments/` for example configs. Any hyperparameter in `config.py` can be overridden in YAML.

### Run tests

```bash
uv run pytest tests/ -v
```

### All arguments

| Argument | Values | Description |
|---|---|---|
| `--model` | `euclidean`, `hyperbolic` | Which manifold to use |
| `--optimizer` | `Adam`, `SGD`, `Radam` | Optimizer (Radam = Riemannian Adam) |
| `--task` | `ganea`, `mircea`, `MNIST` | Task type |
| `--loss` | `cross`, `mse` | Loss function |
| `--dataset` | `10`, `30`, `50` | Prefix length (ganea only) |
| `--replace` | float (default 0.5) | Noise fraction for ganea prefix |
| `--runs` | int (default 10) | Number of repeated independent runs |
| `--wandb_project` | str | W&B project name |
| `--debug` | flag | Enable autograd anomaly detection |
| `--config` | path | YAML experiment config file |
| `--filter` | `key=val,...` | Filter sweep combinations |
| `--experiment` | int | Run single experiment by index |
| `--dry-run` | flag | List experiments without running |

## TODO

- [x] Add more datasets
- [x] Implement Riemannian Adam
- [x] Add test suite
- [ ] Implement Riemannian SGD
- [ ] Expand to RNN

## References

- Hyperbolic Neural Networks: [paper](https://arxiv.org/abs/1805.09112) · [code](https://github.com/dalab/hyperbolic_nn)
- Poincaré Embeddings: [paper](https://papers.nips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html) · [code](https://github.com/facebookresearch/poincare-embeddings)
- Fully Hyperbolic NN: [code](https://github.com/chenweize1998/fully-hyperbolic-nn/tree/main/gcn)
- [Hyperbolic Learning Algorithms](https://github.com/drewwilimitis/hyperbolic-learning)
