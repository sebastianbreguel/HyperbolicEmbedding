import pandas as pd
import numpy as np
from time import perf_counter
from random import choice, sample, seed, random

from .data_Params import (
    NG,
    LARGE,
    NUMBERS,
    URL_PREFIX_50,
    URL_PREFIX_30,
    URL_PREFIX_10,
    MAX_RANDOM,
    MIN_RANDOM,
    ROUND,
    VOCABULARY,
    EMB,
    NM,
    V,
    URL,
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


def data_ganea(replace, positivi):
    seed(SEED)
    num = NUMBERS

    lista_50 = [[]] * NG
    lista_30 = [[]] * NG
    lista_10 = [[]] * NG

    total = 20 * LARGE
    porcent_50 = int(total * 0.5)
    porcent_30 = int(total * 0.3)
    porcent_10 = int(total * 0.1)
    K1 = int(porcent_50 * 0.5)
    K2 = int(porcent_30 * 0.5)
    K3 = int(porcent_10 * 0.5)

    positives = 0
    negatives = 0
    for i in range(NG):

        a = ""
        for _ in range(total):
            a += choice(num)

        b = ""
        d = ""
        e = ""
        if random() < positivi:
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
            lista_50[i] = [a, b, 0, 1]
            lista_30[i] = [a, d, 0, 1]
            lista_10[i] = [a, e, 0, 1]

        else:
            negatives += 1
            for _ in range(porcent_50):
                b += choice(num)

            for _ in range(porcent_30):
                d += choice(num)

            for _ in range(porcent_10):
                e += choice(num)

            lista_50[i] = [a, b, 1, 0]
            lista_30[i] = [a, d, 1, 0]
            lista_10[i] = [a, e, 1, 0]
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
