"""Dataset generation utilities for the three experimental tasks.

Tasks
-----
ganea  : Binary prefix-classification dataset (Ganea et al. word embeddings).
         Generates CSV files with character-level one-hot-ish encodings.
mircea : Phylogenetic tree task with 7 nodes (a→b,c→d,e,f,g) and Hamming distances.
MNIST  : Downloads MNIST, reduces to ``DIMENSIONS`` via UMAP, and saves CSVs.
"""

from __future__ import annotations

import os
from random import choice, random, sample, seed
from time import sleep
from typing import List

import pandas as pd
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import umap
from config import (
    DIMENSIONS, EMB, LARGE, MAX_RANDOM, MIN_RANDOM,
    NGS, NM, NUMBERS, POSITIVE, ROUND, SEED, URL,
    URL_PREFIX_10, URL_PREFIX_30, URL_PREFIX_50,
    VOCABULARY, WORD_LARGE, WORDS, V,
)


def generate_data(create_folder: bool, replace: float, task: str) -> None:
    """Dispatch to the task-specific data generator.

    Args:
        create_folder: Whether to create the ``data/Prefix`` directory first.
        replace: Fraction of characters replaced in the ganea prefix task.
        task: One of ``"ganea"``, ``"mircea"``, ``"MNIST"``.
    """
    if create_folder:
        print("Creating folder")
        os.makedirs("data/Prefix", exist_ok=True)
        sleep(1)
        print("\nFolder created\n")

    if task == "ganea":
        data_ganea(replace)
    elif task == "mircea":
        data_mircea()
    elif task == "MNIST":
        data_MNIST()


def prefixWord(porcent: int, replaced: int, previus: str) -> str:
    """Create a noisy prefix by randomly replacing ``replaced`` positions.

    Args:
        porcent: Total prefix length.
        replaced: Number of characters to replace with random noise.
        previus: The original word to derive the prefix from.

    Returns:
        A string of length ``porcent`` with ``replaced`` random substitutions.
    """
    new_word = ""
    r = sample(range(porcent), replaced)
    for i in range(porcent):
        if i in r:
            new_word += choice(NUMBERS)
        else:
            new_word += previus[i]
    return new_word


def generate_df(porcentaje: float, url: str, replace: float, words_bank: List[str]) -> None:
    """Generate and save a ganea-style prefix CSV at the given prefix fraction.

    Args:
        porcentaje: Fraction of ``WORD_LARGE`` characters used as the prefix.
        url: Output CSV path (without extension).
        replace: Noise fraction for ``prefixWord``.
        words_bank: Pool of base words to sample from.
    """
    lista = [[]] * NGS
    porcent = int(porcentaje * WORD_LARGE)
    K = int(porcent * replace)
    print(f"{K}-{porcent}")

    for i in range(NGS):
        a = choice(words_bank)
        if random() < POSITIVE:
            b = prefixWord(porcent, K, a)
            lista[i] = [a, b, 0, 1]
        else:
            b = "".join(sample(NUMBERS, porcent))
            lista[i] = [a, b, 1, 0]

    df = pd.DataFrame(lista, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"])
    for i in range(WORD_LARGE * LARGE):
        df[f"word-{i}"] = df["Word"].apply(lambda x: str(x)[i])
    for i in range(porcent):
        df[f"prefix-{i}"] = df["Prefix"].apply(lambda x: str(x)[i])

    df = df.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df.to_csv(url)


def data_ganea(replace: float) -> None:
    """Generate the ganea prefix task CSVs (10%, 30%, 50% prefix lengths).

    Args:
        replace: Fraction of prefix characters replaced with random noise.
    """
    seed(SEED)
    words_bank: List[str] = []
    for _ in range(WORDS):
        words_bank.append("".join(sample(NUMBERS, WORD_LARGE * LARGE)))

    generate_df(0.5, f"{URL_PREFIX_50}_{replace}.csv", replace, words_bank)
    generate_df(0.3, f"{URL_PREFIX_30}_{replace}.csv", replace, words_bank)
    generate_df(0.1, f"{URL_PREFIX_10}_{replace}.csv", replace, words_bank)


def data_mircea() -> None:
    """Generate the mircea phylogenetic tree dataset and save to ``URL``.

    Builds a balanced binary tree with 7 nodes::

           A
         B   C
        D E F G

    Each sample contains 7 nodes encoded as integer indices (via ``EMB``) and
    6 pairwise mutation probabilities as regression targets.
    """
    import random as rng

    lista = []
    for _ in range(NM):
        p1 = round(rng.uniform(MIN_RANDOM, MAX_RANDOM), ROUND)
        p2 = round(rng.uniform(MIN_RANDOM, MAX_RANDOM), ROUND)
        p3 = round(rng.uniform(MIN_RANDOM, MAX_RANDOM), ROUND)
        p4 = round(rng.uniform(MIN_RANDOM, MAX_RANDOM), ROUND)
        p5 = round(rng.uniform(MIN_RANDOM, MAX_RANDOM), ROUND)
        p6 = round(rng.uniform(MIN_RANDOM, MAX_RANDOM), ROUND)

        lista.append([])
        a = [choice(VOCABULARY) for _ in range(V)]

        def mutate(parent, p):
            return [choice(VOCABULARY) if rng.random() < p else parent[i] for i in range(V)]

        b = mutate(a, p1)
        c = mutate(a, p2)
        d = mutate(b, p3)
        e = mutate(b, p4)
        f = mutate(c, p5)
        g = mutate(c, p6)

        for node in [a, b, c, d, e, f, g]:
            for i in range(V):
                node[i] = EMB[node[i]]

        dist = [p1, p2, p3, p4, p5, p6]
        for node in [a, b, c, d, e, f, g]:
            lista[-1].extend(node)
        lista[-1].extend(dist)

    df = pd.DataFrame(lista)
    df.to_csv(URL)


def data_MNIST() -> None:
    """Download MNIST, reduce to ``DIMENSIONS`` with UMAP, and save CSVs.

    Saves reduced train/test embeddings to ``data/MNIST/train.csv`` and
    ``data/MNIST/test.csv``.  The UMAP reducer is fit on the training set
    and applied (transform-only) to the test set to avoid leakage.
    """
    train_dataset = dsets.MNIST(root="./", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root="./", train=False, transform=transforms.ToTensor())

    X_train = train_dataset.data.numpy()
    reducer = umap.UMAP(random_state=42, n_components=DIMENSIONS)
    X_train_embed = torch.from_numpy(reducer.fit_transform(pd.DataFrame(X_train.reshape(X_train.shape[0], -1))))
    train_dataset.data = X_train_embed

    X_test = test_dataset.data.numpy()
    X_test_embed = torch.from_numpy(reducer.transform(pd.DataFrame(X_test.reshape(X_test.shape[0], -1))))
    test_dataset.data = X_test_embed

    os.makedirs("data/MNIST", exist_ok=True)
    pd.DataFrame(X_train_embed).to_csv("data/MNIST/train.csv", index=False)
    pd.DataFrame(X_test_embed).to_csv("data/MNIST/test.csv", index=False)
