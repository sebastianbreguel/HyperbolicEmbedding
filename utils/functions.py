import pandas as pd
import numpy as np
from time import perf_counter
from random import choice, sample, seed, random

from .data_Params import (
    NG,
    LARGE,
    NUMBERS,
    URL_PREFIX_50,
    URL_PREFIX_40,
    URL_PREFIX_30,
    URL_PREFIX_20,
    URL_PREFIX_10,
    MAX_RANDOM,
    MIN_RANDOM,
    ROUND,
    VOCABULARY,
    EMB,
    NM,
    V,
    URL,
    WORD_LARGE,
)
from parameters import SEED


def beaty_print(start, initial, value, total, needed):
    end = perf_counter()
    arrows = int(value / total * 10 // 1)
    arrows = "-" + "-" * (arrows) + ">" + " " * (10 - arrows)
    if needed:
        print(f"\n\nLAST: {(end - start):.2f} seconds")
        start = perf_counter()
    stringa = f"{int(value)}/{int(total)}"
    print(
        "{:>17}".format(stringa)
        + f"{arrows}"
        + "{:>5}".format(((value / total * 1000) // 1) / 10)
        + f"% |total time:{((end - initial)/60):.2f} minutes"
    )
    return start


def prefixWord(porcent, K, previus):
    new_word = ""

    r = sample(range(porcent), K)
    for i in range(porcent):
        if i in r:
            new_word += choice(NUMBERS)
        else:
            new_word += previus[i]
    return new_word


def processdf(porcentaje, df, url):
    for i in range(WORD_LARGE):
        df[f"word-{i}"] = df["Word"].apply(lambda x: str(x)[i])
        if i < porcentaje:
            df[f"prefix-{i}"] = df["Prefix"].apply(lambda x: str(x)[i])

    df = df.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df.to_csv(url)


def generate_df(porcentaje, url, positivi, replace, word_bank):

    lista = [[]] * NG
    # amount of the random/Prefix
    porcent = int(porcentaje * WORD_LARGE)
    # letter to change
    K = int(porcent * replace)

    for i in range(NG):
        a = word_bank[i]
        if random() < positivi:
            b = prefixWord(porcent, K, a)
            lista[i] = [a, b, 0, 1]

        else:
            b = "".join(sample(NUMBERS, porcent))
            lista[i] = [a, b, 1, 0]

    df = pd.DataFrame(lista, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"])
    processdf(porcent, df, url)


def data_ganea(replace: float, positivi: float) -> None:
    seed(SEED)
    word_bank = []
    for _ in range(NG):
        word_bank.append("".join(sample(NUMBERS, WORD_LARGE)))

    generate_df(0.5, URL_PREFIX_50, positivi, replace, word_bank)
    generate_df(0.4, URL_PREFIX_40, positivi, replace, word_bank)
    generate_df(0.3, URL_PREFIX_30, positivi, replace, word_bank)
    generate_df(0.2, URL_PREFIX_20, positivi, replace, word_bank)
    generate_df(0.1, URL_PREFIX_10, positivi, replace, word_bank)


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
