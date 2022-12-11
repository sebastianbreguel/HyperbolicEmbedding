# EmbeddingHyperbolic

This is a repository to investigate how to generate music based in Hyperbolic Embeddings

```bash
main.py
|
|__📂utils                              # utils function to create, get, process and run data/metrics/models
|   |__📜model_data.py
|   |__📜parameters.py
|   |__📜run.py
|   |__📜stadistic_util.py
|   |__📜train_functions.py
|
|__📂manifolds                          # Manifolds to use in the project
|   |__📜base.py
|   |__📜Euclidean.py
|   |__📜math_util.py
|   |__📜poincare.py
|
|__📂layers                             # Manifold Layers
|   |__📜layers.py
|   |__📜hyp_layers.py
|   |__📜hyp_softmax.py
|
|__📂models                             # NN for the project
|   |__ ganeaPrefix.py
|
|__📂Optimizer
|   |__ 📜Radam.py
|
|__📂data                               # Data to use in the project
|   |__📜data_gen.py
|   |__📜data_main.py
|
|_📂analisis                            # jupyter notebooks where we analize the results
|  |
|  |__📂 Ganea
|  |   |__📂First
|  |   |   |__📜analisis.ipynb
|  |   |
|  |   |
|  |   |__📂Second
|  |   |   |__📜analisis2.ipynb

```

# Usage

## Arguments to create data

```bash
--generate_data : Generate data from midi files
--create_folder : Create folder to save data
--task: Task to train, could be ganea/MNIST(classification) or mircea(regression)
--replace: if task 'ganea'-> Make prefix with noise
```

## Arguments to run model

```bash
--train_eval : Train and evaluate the model
--model: Model to train, colud be euclidean or hyperbolic
--optimizer: Optimizer to train, could be SGD or Adam
--loss: Loss to train, could be mse or cross
--dataset: determinate the large of the prefix
        - 0: if task mircea
        - 10, 30 o 50: if task ganea
```

### Examples

1: Run model euclidean with dataset 10 on task ganea

```python
python main.py --train_eval --model euclidean --task ganea --loss cross --dataset 10

```

2: Run model hyperbolic with dataset 10 on task ganea

```python
python main.py --train_eval --model hyperbolic --task ganea --loss cross --dataset 10
```

3: Run model euclidean with dataset 0 on task ganea

```python
python main.py --train_eval --model euclidean --task mircea --loss mse --dataset 0

```

4: **_run and create data_**

```python
python main.py --generate_data --create_folder --task ganea --replace --train_eval --model euclidean --task ganea --loss cross --dataset 10
```

#### TODO

- [x] Add more datasets
- [x] Implement Riemannian Adam
- [ ] Implement Riemannian SGD
- [ ] Expand to RNN

## References

- Hyperbolic Neural Networks: [paper](https://arxiv.org/abs/1805.09112)-[github](https://github.com/dalab/hyperbolic_nn)
- Codes for Network Embedding: [github](https://github.com/chenweize1998/fully-hyperbolic-nn/tree/main/gcn)
- Poincaré Embeddings for Learning Hierarchical Representations: [paper](https://papers.nips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html) - [github](https://github.com/facebookresearch/poincare-embeddings)

## Usefull

- [Hyperbolic Algorithms](https://github.com/drewwilimitis/hyperbolic-learning)
- [Hyperbolic Embedding](https://github.com/prokopevaleksey/poincare-embeddings)
