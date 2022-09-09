import pandas as pd
import numpy as np
from parameters import *

# funcion of the real distance in a tree between two nodes
def metric(
    x,
    y,
):
    h = 0
    for i in range(min(len(x), len(y))):
        if x[i] == y[i]:
            h += 1
        else:
            break
    return len(x) + len(y) - 2 * h


# function to generate a random word
def generate_word(max_length=L, vocabuary=["a", "b", "c"]):
    s = ""
    # print("maxlength", max_length)
    number = np.random.randint(0, max_length + 1)
    for _ in range(number):
        s += np.random.choice(vocabuary)
    return s


def embedding1(x):
    l = 0
    for i in range(len(x)):
        l += (i + 1) * (S.find(x[i]) + 1)
    return np.array(l)


def embedding2(x):
    l = 0
    for i in range(len(x)):
        l += S.find(x[i]) + 1
    return np.array(l)


def embed(x):
    l = np.zeros(L)
    for i in range(len(x)):
        l[i] = S.find(x[i])
    return np.array(l)
