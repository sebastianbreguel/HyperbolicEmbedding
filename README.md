# EmbeddingHyperbolic

This is a repository to investigate how to generate music based in Hyperbolic Embeddings

To create the dataset :

```sh
python3 main.py --gen_data
```

To run the model:

- Euclidean Adam:

```sh
python3 main.py --make_train_eval --model euclidean --optimizer Adam
```

```sh
python main.py --make_train_eval --model euclidean --optimizer Adam
```

- Euclidean RiemannianAdam:

```sh
python3 main.py --make_train_eval --model euclidean --optimizer RiemannianAdam
```

```sh
python main.py --make_train_eval --model euclidean --optimizer RiemannianAdam
```

- Hyperbolic Adam:

```sh
python3 main.py --make_train_eval --model hyperbolic --optimizer Adam
```

```sh
python main.py --make_train_eval --model hyperbolic --optimizer Adam
```

- Hyperbolic RiemannianAdam:

```sh
python3 main.py --make_train_eval --model hyperbolic --optimizer RiemannianAdam
```

```sh
python main.py --make_train_eval --model hyperbolic --optimizer RiemannianAdam
```
