import pandas as pd
import numpy as np
from time import perf_counter
from random import choice, random, sample

from data_Params import *
from functions import *

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
            end = perf_counter()
            arrows = int(value / total * 10 // 1)
            arrows = "-" + "-" * (arrows) + ">" + " " * (10 - arrows)
            print(f"\n\nLAST: {(end - start):.2f} seconds")
            start = perf_counter()
            print(
                f"{int(value/1000)}/{int(total/1000)}{arrows}{((value/total * 1000)//1)/10}% \n|total time:{((end - initial)/60):.2f} minutes\n"
            )

    end = perf_counter()
    arrows = "-" + "-" * (10) + ">"
    print(f"\n\nLAST: {(end - start):.2f} seconds")
    print(
        f"{int(value/1000)}/{int(total/1000)}{arrows}{((value/total * 1000)//1)/10}% \n|total time:{((end - initial)/60):.2f} minutes\n"
    )

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

    for _ in range(NG):

        a = ""
        for _ in range(20):
            a += choice(num)

        b = ""
        if random() > R:
            c = 1
            r = sample(a, K)
            for i in range(15):
                if i in r:
                    b += choice(num)
                else:
                    b += a[i]
        else:
            c = 0
            for _ in range(15):
                b += choice(num)

        d = pd.DataFrame([[int(a), int(b), c]], columns=list("ABC"))
        df = pd.concat([df, d], ignore_index=True)
    df.to_csv(URL_GANEA)
    # print(df)
# data_ganea()
