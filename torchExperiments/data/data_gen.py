import os
from random import choice, random, sample, seed
from time import sleep

import pandas as pd
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import umap
from utils.parameters import (DIMENTIONS, EMB, LARGE, MAX_RANDOM, MIN_RANDOM,
                              NGS, NM, NUMBERS, POSITIVE, ROUND, SEED, URL,
                              URL_PREFIX_10, URL_PREFIX_30, URL_PREFIX_50,
                              VOCABULARY, WORD_LARGE, WORDS, V)


def generate_data(create_folder, replace, task) -> None:
    if create_folder:
        print("Creating folder")
        os.system("mkdir Prefix")
        sleep(1)
        print("\nFolder created\n")

    if task == "ganea":
        data_ganea(replace)

    elif task == "mircea":
        data_mircea()

    elif task == "MNIST":
        data_MNIST()


def prefixWord(porcent, replaced, previus):
    new_word = ""

    r = sample(range(porcent), replaced)

    for i in range(porcent):
        if i in r:
            new_word += choice(NUMBERS)
        else:
            new_word += previus[i]
    return new_word


def generate_df(porcentaje, url, replace, words_bank):
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
    print(df)
    print(LARGE, NGS)
    for i in range(WORD_LARGE * LARGE):
        df[f"word-{i}"] = df["Word"].apply(lambda x: str(x)[i])

    for i in range(porcent):
        df[f"prefix-{i}"] = df["Prefix"].apply(lambda x: str(x)[i])

    df = df.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    print(df)
    #  removde "data/" from url
    df.to_csv(url[5:])


def data_ganea(replace: float) -> None:
    seed(SEED)
    words_bank = []
    print(LARGE, WORDS)
    for _ in range(WORDS):
        words_bank.append("".join(sample(NUMBERS, WORD_LARGE * LARGE)))

    generate_df(0.5, f"{URL_PREFIX_50}_{replace}.csv", replace, words_bank)
    generate_df(0.3, f"{URL_PREFIX_30}_{replace}.csv", replace, words_bank)
    generate_df(0.1, f"{URL_PREFIX_10}_{replace}.csv", replace, words_bank)


def data_mircea():
    #    A
    #  B   C
    # D E F G

    lista = []
    for _ in range(NM):
        p1 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), ROUND)
        p2 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), ROUND)
        p3 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), ROUND)
        p4 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), ROUND)
        p5 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), ROUND)
        p6 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), ROUND)
        lista.append([])
        a = []
        for _ in range(V):
            a.append(choice(VOCABULARY))

        b = []
        for i in range(V):
            r = random.random()
            if r < p1:
                b.append(choice(VOCABULARY))
            else:
                b.append(a[i])

        c = []
        for i in range(V):
            r = random.random()
            if r < p2:
                c.append(choice(VOCABULARY))
            else:
                c.append(a[i])

        d = []
        for i in range(V):
            r = random.random()
            if r < p3:
                d.append(choice(VOCABULARY))
            else:
                d.append(b[i])

        e = []
        for i in range(V):
            r = random.random()
            if r < p4:
                e.append(choice(VOCABULARY))
            else:
                e.append(b[i])

        f = []
        for i in range(V):
            r = random.random()
            if r < p5:
                f.append(choice(VOCABULARY))
            else:
                f.append(c[i])

        g = []
        for i in range(V):
            r = random.random()
            if r < p6:
                g.append(choice(VOCABULARY))
            else:
                g.append(c[i])

        for i in range(V):
            a[i] = EMB[a[i]]
            b[i] = EMB[b[i]]
            c[i] = EMB[c[i]]
            d[i] = EMB[d[i]]
            e[i] = EMB[e[i]]
            f[i] = EMB[f[i]]
            g[i] = EMB[g[i]]

        dist = [
            p1,
            p2,
            p1 + p3,
            p1 + p4,
            p2 + p5,
            p2 + p6,
            p1 + p2,
            p3,
            p4,
            p1 + p2 + p5,
            p1 + p2 + p6,
            p2 + p1 + p3,
            p2 + p1 + p4,
            p5,
            p6,
            p3 + p4,
            p3 + p1 + p2 + p5,
            p3 + p1 + p2 + p6,
            p4 + p1 + p2 + p5,
            p4 + p1 + p2 + p6,
            p5 + p6,
        ]
        dist = [p1, p2, p3, p4, p5, p6]

        for i in a:
            lista[-1].append(i)
        for i in b:
            lista[-1].append(i)
        for i in c:
            lista[-1].append(i)
        for i in d:
            lista[-1].append(i)
        for i in e:
            lista[-1].append(i)
        for i in f:
            lista[-1].append(i)
        for i in g:
            lista[-1].append(i)
        for i in dist:
            lista[-1].append(i)

    df = pd.DataFrame(lista)
    df.to_csv(URL)


def data_MNIST():
    train_dataset = dsets.MNIST(
        root="./", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = dsets.MNIST(root="./", train=False, transform=transforms.ToTensor())

    # * EMBEDDING
    X_train = train_dataset.data.numpy()
    df = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
    reducer = umap.UMAP(random_state=42, n_components=DIMENTIONS)
    X_train = torch.from_numpy(reducer.fit_transform(df))
    train_dataset.data = X_train

    X_test = test_dataset.data.numpy()
    df = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
    X_test = torch.from_numpy(reducer.transform(df))
    test_dataset.data = X_test

    # * TOCSV
    pd.DataFrame(X_train).to_csv("MNIST/train.csv", index=False)
    pd.DataFrame(X_test).to_csv("MNIST/test.csv", index=False)
