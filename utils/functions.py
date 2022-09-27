import pandas as pd
import numpy as np
from time import perf_counter
from random import choice, random, sample, seed

from .data_Params import *


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


def data_ganea(replace):
    seed(1862554)
    num = NUMBERS
    print(replace)

    lista_50 = [[]] * NG
    lista_30 = [[]] * NG
    lista_10 = [[]] * NG

    total = 20 * LARGE
    porcent_50 = int(total * 0.5)
    porcent_30 = int(total * 0.3)
    porcent_10 = int(total * 0.1)

    print(f"Total: {total} | 50%: {porcent_50} | 30%: {porcent_30} | 10%: {porcent_10}")
    print(f"K -> 50%: {K+2} | 30%: {K} | 10%:{K/3}")

    positives = 0
    negatives = 0
    for i in range(NG):

        a = ""
        for _ in range(total):
            a += choice(num)

        b = ""
        d = ""
        e = ""

        if random() < R:
            positives += 1

            # CHANGES OF LETTERS IN THE PREFIX
            r = sample(range(porcent_50), K+2)
            for z in range(porcent_50):
                if z in r and replace:
                    b += choice(num)
                else:
                    b += a[z]

            r = sample(range(porcent_30), int(K ))
            for z in range(porcent_30):
                if z in r and replace:
                    d += choice(num)
                else:
                    d += a[z]

            r = sample(range(porcent_10), int(K / 3))
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
