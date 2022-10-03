# EmbeddingHyperbolic

This is a repository to investigate how to generate music based in Hyperbolic Embeddings

```bash
main.py
|__📂utils
|   |__📜functions.py
|   |__📜generate.py
|   |__📜model_data.py
|   |__📜parameters.py
|
|__📂NNs
|   |__ 📜hyperbolic.py      #have the neuronal network for both manfiolds
|
|__📂Optimizer
|   |__ 📜Radam.py           #not ready
|
|__📂Manifolds
|   |__📜base.py
|   |__📜Euclidean.py
|   |__📜poincare.py
|
|__📂data
|   |__📜data_10.csv
|   |__📜data_30.csv
|   |__📜data_50.csv
|   |__📜Phylogenetics.csv    # For mircea experiment
|
|_📂Analisis
   |__📜Analisis.ipynb

```

# Usage

#### Arguments to create data

```bash
--generate_data : Generate data from midi files
--create_folder : Create folder to save data
--task: Task to train, could be ganea(classification) or mircea(regression)
--replace: if task 'ganea'-> Make prefix with noise
```

#### Arguments to run model

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

1. Run model euclidean with dataset 10 on task ganea

```python
python main.py --train_eval --model euclidean --task ganea --loss cross --dataset 10

```

2. Run model hyperbolic with dataset 10 on task ganea

```python
python main.py --train_eval --model hyperbolic --task ganea --loss cross --dataset 10
```

3. Run model euclidean with dataset 0 on task ganea

```python
python main.py --train_eval --model euclidean --task mircea --loss mse --dataset 0

```

4. **_run and create data_**

```python
python main.py --generate_data --create_folder --task ganea --replace --train_eval --model euclidean --task ganea --loss cross --dataset 10
```

#### TODO

- [ ] Add more datasets
- [ ] Implement Riemannian Adam
- [ ] Implement Riemannian SGD
- [ ] Expand to RNN
