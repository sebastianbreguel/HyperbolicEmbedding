import os
from time import sleep
import pandas as pd
import numpy as np
from time import perf_counter
from random import choice, sample, seed, random
from .parameters import (
    NG,
    LARGE,
    NUMBERS,
    URL_PREFIX_50,
    URL_PREFIX_30,
    URL_PREFIX_10,
    MAX_RANDOM,
    MIN_RANDOM,
    ROUND,
    POSITIVE,
    VOCABULARY,
    EMB,
    NM,
    V,
    SEED,
    URL,
    WORD_LARGE,
)


def generate_data(create_folder, replace, task) -> None:
    # Generate the folder

    if create_folder:
        print("Creating folder")
        os.system("mkdir data")
        os.system("mkdir data/Prefix")
        sleep(1)
        print("\nFolder created\n")
    # run generate_data.py
    if task == "ganea":

        data_ganea(replace)

    elif task == "mircea":
        data_mircea()

    print("# Data Generated #")


def prefixWord(porcent, replaced, previus):
    new_word = ""

    r = sample(range(porcent), replaced)
    for i in range(porcent):
        if i in r:
            new_word += choice(NUMBERS)
        else:
            new_word += previus[i]
    return new_word


def processdf(porcentaje, df, url):
    for i in range(WORD_LARGE):
        df[f"word-{i}"] = df["Word"].apply(lambda x: str(x)[i])

    for i in range(porcentaje):
        df[f"prefix-{i}"] = df["Prefix"].apply(lambda x: str(x)[i])

    df = df.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df.to_csv(url)


def generate_df(porcentaje, url, replace, words_bank):

    lista = [[]] * NG
    # amount of the random/Prefix
    porcent = int(porcentaje * WORD_LARGE)
    # letter to change
    K = int(porcent * replace)
    print(f"{K}-{porcent}")

    for i in range(NG):
        a = words_bank[i]
        if random() < POSITIVE:
            b = prefixWord(porcent, K, a)
            lista[i] = [a, b, 0, 1]

        else:
            b = "".join(sample(NUMBERS, porcent))
            lista[i] = [a, b, 1, 0]

    df = pd.DataFrame(lista, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"])
    processdf(porcent, df, url)


def data_ganea(replace: float) -> None:
    seed(SEED)
    words_bank = []
    for _ in range(NG):
        words_bank.append("".join(sample(NUMBERS, WORD_LARGE)))

    generate_df(0.5, f"{URL_PREFIX_50}_{replace}.csv", replace, words_bank)
    generate_df(0.3, f"{URL_PREFIX_30}_{replace}.csv", replace, words_bank)
    generate_df(0.1, f"{URL_PREFIX_10}_{replace}.csv", replace, words_bank)


def data_ganea1(replace, Positives):
    num = NUMBERS

    lista_50 = [[]] * NG
    lista_30 = [[]] * NG
    lista_10 = [[]] * NG

    total = 20 * LARGE
    porcent_50 = int(total * 0.5)
    porcent_30 = int(total * 0.3)
    porcent_10 = int(total * 0.1)
    K1 = int(porcent_50 * replace)
    K2 = int(porcent_30 * replace)
    K3 = int(porcent_10 * replace)

    print(f"Total: {total} | 50%: {porcent_50} | 30%: {porcent_30} | 10%: {porcent_10}")
    positives = 0
    negatives = 0
    for i in range(NG):

        a = ""
        for _ in range(total):
            a += choice(num)

        b = ""
        d = ""
        e = ""
        if random() < Positives:
            positives += 1

            # CHANGES OF LETTERS IN THE PREFIX
            r = sample(range(porcent_50), K1)
            for z in range(porcent_50):
                if z in r and replace:
                    b += choice(num)
                else:
                    b += a[z]
            r = sample(range(porcent_30), K2)
            for z in range(porcent_30):
                if z in r and replace:
                    d += choice(num)
                else:
                    d += a[z]
            r = sample(range(porcent_10), K3)
            for z in range(porcent_10):
                if z in r and replace:
                    e += choice(num)
                else:
                    e += a[z]
            #
            lista_50[i] = [a, b, 1, 0]
            lista_30[i] = [a, d, 1, 0]
            lista_10[i] = [a, e, 1, 0]

        else:
            negatives += 1
            for _ in range(porcent_50):
                b += choice(num)

            for _ in range(porcent_30):
                d += choice(num)

            for _ in range(porcent_10):
                e += choice(num)

            lista_50[i] = [a, b, 0, 1]
            lista_30[i] = [a, d, 0, 1]
            lista_10[i] = [a, e, 0, 1]
    print(f"Positives: {positives} | Negatives: {negatives}")

    df_50 = pd.DataFrame(
        lista_50, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"]
    )
    df_30 = pd.DataFrame(
        lista_30, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"]
    )
    df_10 = pd.DataFrame(
        lista_10, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"]
    )

    for i in range(total):
        df_50[f"word-{i}"] = df_50["Word"].apply(lambda x: str(x)[i])
        df_30[f"word-{i}"] = df_30["Word"].apply(lambda x: str(x)[i])
        df_10[f"word-{i}"] = df_10["Word"].apply(lambda x: str(x)[i])

    for i in range(porcent_50):
        df_50[f"prefix-{i}"] = df_50["Prefix"].apply(lambda x: str(x)[i])

    for i in range(porcent_30):
        df_30[f"prefix-{i}"] = df_30["Prefix"].apply(lambda x: str(x)[i])

    for i in range(porcent_10):
        df_10[f"prefix-{i}"] = df_10["Prefix"].apply(lambda x: str(x)[i])

    df_50 = df_50.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df_50.to_csv(URL_PREFIX_50)

    df_30 = df_30.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df_30.to_csv(URL_PREFIX_30)

    df_10 = df_10.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df_10.to_csv(URL_PREFIX_10)


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
