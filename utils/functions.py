import random
import torch
import pandas as pd
import numpy as np
import csv

V = 5
L = 15
N = 1000

s = "abcdefghijklmnopqrstuvwxyz"[:V]
vocabuary = [i for i in s]


def metric(x, y) -> int:
    h = 0
    max = min(len(x), len(y))
    while h < max and x[h] == y[h]:
        h += 1
    return len(x) + len(y) - 2 * h


def generate_word(max_length=20, vocabuary=["a", "b", "c"]) -> str:
    s = ""
    length = random.randint(1, max_length)
    for _ in range(length):
        s += random.choice(vocabuary)
    return s


words = [" "] * N
for _ in range(N):
    a = generate_word(L, vocabuary)
    words[_] = a

df_words = pd.DataFrame(words, columns=["word"]).drop_duplicates()
# reset the index of the dataframe
df_words = df_words.reset_index(drop=True)
n = len(df_words)
lista = [[]] * (int((n - 1) * (n) / 2))

value = 0
for i in range(n):
    a = df_words.iloc[i]["word"]

    for _ in range(i + 1, n):
        # select two random words and then aply metric

        b = df_words.iloc[_]["word"]

        lista[value] = [a, b, metric(a, b)]
        value += 1

# dataframe of the words and the distance
df_distances = pd.DataFrame(lista, columns=["words 1", "words 2", "metric"])

df_words.to_csv(f"data/tree_{V}_{L}.csv")
df_distances.to_csv(f"data/distance_{V}_{L}.csv")
