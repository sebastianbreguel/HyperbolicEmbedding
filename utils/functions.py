import pandas as pd
import numpy as np
from time import perf_counter
from random import choice, random, sample, randrange

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


def data_ganea(num=NUMBERS):

    initial = perf_counter()
    start = initial

    lista_15 = [[]] * NG
    lista_10 = [[]] * NG
    lista_05 = [[]] * NG

    large = 1
    total = 20 * large
    porcent_15 = 15 * large
    porcent_10 = 10 * large
    porcent_05 = 5 * large

    for i in range(NG):

        a = ""
        if i % (NG / 10) == 0:
            start = beaty_print(start, initial, i, NG, False)
        for j in range(total):
            a += choice(num)

        b = ""
        d = ""
        e = ""

        if random() < R:
            r = sample(range(porcent_15), K)
            for z in range(porcent_15):
                if z in r:
                    b += choice(num)
                else:
                    b += a[z]
            lista_15[i] = [a, b, 1, 0]

            r = sample(range(porcent_10), K)
            for z in range(porcent_10):
                if z in r:
                    b += choice(num)
                else:
                    b += a[z]
            lista_10[i] = [a, b, 1, 0]

            r = sample(range(porcent_05), K)
            for z in range(porcent_05):
                if z in r:
                    b += choice(num)
                else:
                    b += a[z]
            lista_05[i] = [a, b, 1, 0]

        else:
            for _ in range(porcent_15):
                b += choice(num)

            for _ in range(porcent_10):
                d += choice(num)

            for _ in range(porcent_05):
                e += choice(num)

            lista_15[i] = [a, b, 0, 1]
            lista_10[i] = [a, d, 0, 1]
            lista_05[i] = [a, e, 0, 1]
    beaty_print(start, initial, NG, NG, False)

    df_15 = pd.DataFrame(
        lista_15, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"]
    )
    df_10 = pd.DataFrame(
        lista_10, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"]
    )
    df_05 = pd.DataFrame(
        lista_05, columns=["Word", "Prefix", "isPrefix", "isNotPrefix"]
    )

    for i in range(total):
        df_15[f"word-{i}"] = df_15["Word"].apply(lambda x: str(x)[i])
        df_10[f"word-{i}"] = df_10["Word"].apply(lambda x: str(x)[i])
        df_05[f"word-{i}"] = df_05["Word"].apply(lambda x: str(x)[i])

    for i in range(porcent_15):
        df_15[f"prefix-{i}"] = df_15["Prefix"].apply(lambda x: str(x)[i])

    for i in range(porcent_10):
        df_10[f"prefix-{i}"] = df_10["Prefix"].apply(lambda x: str(x)[i])

    for i in range(porcent_05):
        df_05[f"prefix-{i}"] = df_05["Prefix"].apply(lambda x: str(x)[i])

    df_15 = df_15.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df_15.to_csv(URL_GANEA_15)

    df_10 = df_10.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df_10.to_csv(URL_GANEA_10)

    df_05 = df_05.drop(["Word", "Prefix"], axis=1).drop_duplicates()
    df_05.to_csv(URL_GANEA_05)
