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


# funcion of the real distance in a tree between two nodes
def metric(x: int, y: int) -> float:
    h = 0
    for i in range(min(len(x), len(y))):
        if x[i] == y[i]:
            h += 1
        else:
            break
    value = len(x) + len(y) - 2 * h
    # print(value/1000)
    return value


# function to generate a random word
def generate_word(max_length: int = L, vocabuary: list = ["a", "b", "c"]) -> str:
    s = ""
    # print("maxlength", max_length)
    number = np.random.randint(0, max_length + 1)
    for _ in range(number):
        s += np.random.choice(vocabuary)
    return s


def embedding1(x: int) -> np.array:
    l = 0
    for i in range(len(x)):
        l += (i + 1) * (S.find(x[i]) + 1)
    return np.array(l)


def embedding2(x: int) -> np.array:
    l = 0
    for i in range(len(x)):
        l += S.find(x[i]) + 1
    return np.array(l)


def embed(x: int) -> np.array:
    l = np.zeros(L)
    for i in range(len(x)):
        l[i] = S.find(x[i])
    return np.array(l)


def generate_words():
    words = [" "] * N
    for _ in range(N):
        a = generate_word(L, VOCABULARY)
        words[_] = a
    df_words = pd.DataFrame(words, columns=["word"]).drop_duplicates()
    # reset the index of the dataframe

    df_words = df_words.reset_index(drop=True)
    n = len(df_words)
    total = int((n - 1) * (n) / 2)

    lista = [[]] * total
    value = 0
    start = perf_counter()
    initial = start
    for i in range(n):
        a = df_words.iloc[i]["word"]

        for j in range(i + 1, n):
            # select two random words and then aply metric

            b = df_words.iloc[j]["word"]

            lista[value] = [a, b, metric(a, b)]
            value += 1
        if value % 100 == 0:
            start = beaty_print(start, initial, value, total, True)

    beaty_print(start, initial, value, total, True)

    # # dataframe of the words and the distance
    df_distances = pd.DataFrame(lista, columns=["words 1", "words 2", "metric"])
    df_distances.to_csv(URL_DISTANCES)
    df_words.to_csv(f"data/words/{V}_{L}.csv")
    print("#" * 29)
    print("Words and distances generated")


def make_embedding():
    print("\nDoing Embedding")
    print("#" * 29)

    start = perf_counter()
    df = pd.read_csv(URL_DISTANCES, header=0)
    df = df.drop(df.columns[0], axis=1)
    df.columns = ["First", "Second", "Metric"]
    df = df.sample(frac=1, random_state=1).reset_index().drop("index", axis=1)

    df.fillna("", inplace=True)

    print(f"\n-Read CSV: {(perf_counter() - start):.2f} seconds")
    df["EF1"] = np.array(df["First"].apply(embedding1))

    print(f"\n-First Embedding: {(perf_counter() - start):.2f} seconds")
    df["EF2"] = np.array(df["First"].apply(embedding2))

    print(f"\n-Second Embedding: {(perf_counter() - start):.2f} seconds")
    df["ES1"] = np.array(df["Second"].apply(embedding1))

    print(f"\n-Third Embedding: {(perf_counter() - start):.2f} seconds")
    df["ES2"] = np.array(df["Second"].apply(embedding2))

    print(f"\n-Forth Embedding: {(perf_counter() - start):.2f} seconds\n")

    data = df[["EF1", "EF2", "ES1", "ES2", "Metric"]]
    data.to_csv(URL_EMBEDDING)

    print(f"\nData Charged: {(perf_counter() - start):.2f} seconds")


def data_ganea(num=NUMBERS):

    df = pd.DataFrame(columns=list("ABC"))
    first_15 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    initial = perf_counter()
    start = initial
    lista_15 = [[]] * NG
    lista_10 = [[]] * NG
    lista_05 = [[]] * NG
    large = 2
    total = 20 * large
    porcent_15 = 15 * large
    porcent_10 = 10 * large
    porcent_05 = 5 * large

    for i in range(NG):

        a = ""
        if i % (NG / 1000) == 0:
            start = beaty_print(start, initial, i, NG, False)
        for j in range(total):
            a += choice(num)

        b = ""
        d = ""
        e = ""
        if random() > R:
            c = 1
            r = sample(range(porcent_15), K)
            for z in range(porcent_15):
                if z in r:
                    b += choice(num)
                else:
                    b += a[z]
            lista_15[i] = [a, b, 0, 1]
            lista_10[i] = [a, b, 0, 1]
            lista_05[i] = [a, b, 0, 1]
        else:
            c = 1
            for _ in range(porcent_15):
                b += choice(num)

            for _ in range(porcent_10):
                d += choice(num)

            for _ in range(porcent_05):
                e += choice(num)

            lista_15[i] = [a, b, 1, 0]
            lista_10[i] = [a, d, 1, 0]
            lista_05[i] = [a, e, 1, 0]
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
    print(df_05)

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
