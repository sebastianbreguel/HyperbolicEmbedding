import numpy as np
import pandas as pd
from .data_Params import *
from .functions import embedding1, embedding2
from time import perf_counter


def make_embedding():
    print("\nDoing Embedding")
    start = perf_counter()
    df = pd.read_csv(URL_DISTANCES, header=0)
    df = df.drop(df.columns[0], axis=1)
    df.columns = ["First", "Second", "Metric"]
    df = df.sample(frac=1, random_state=1).reset_index().drop("index", axis=1)

    df.fillna("", inplace=True)

    print(f"\nRead CSV: {(perf_counter() - start):.2f} seconds")
    df["EF1"] = np.array(df["First"].apply(embedding1))

    print(f"\nFirst Embedding: {(perf_counter() - start):.2f} seconds")
    df["EF2"] = np.array(df["First"].apply(embedding2))

    print(f"\nSecond Embedding: {(perf_counter() - start):.2f} seconds")
    df["ES1"] = np.array(df["Second"].apply(embedding1))

    print(f"\nThird Embedding: {(perf_counter() - start):.2f} seconds")
    df["ES2"] = np.array(df["Second"].apply(embedding2))

    print(f"\nForth Embedding: {(perf_counter() - start):.2f} seconds\n")

    data = df[["EF1", "EF2", "ES1", "ES2", "Metric"]]
    data.to_csv(URL_EMBEDDING)
    print(URL_EMBEDDING)

    print(data)
    print(f"\nData Charged: {(perf_counter() - start):.2f} seconds")
