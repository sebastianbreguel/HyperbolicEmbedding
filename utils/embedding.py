import numpy as np
import pandas as pd
from parameters import *
from functions import embedding1, embedding2
from time import perf_counter

start = perf_counter()
df = pd.read_csv(URL, header=0)
df = df.drop(df.columns[0], axis=1)
df.columns = ["First", "Second", "Metric"]
df = df.sample(frac=1, random_state=1).reset_index().drop("index", axis=1)

df.fillna("", inplace=True)
# if not FULLY_EUCLIDEAN:
#     pass
print(perf_counter() - start)
df["EF1"] = np.array(df["First"].apply(embedding1))
print("first ready")
print(perf_counter() - start)
df["EF2"] = np.array(df["First"].apply(embedding2))
print("second ready")
print(perf_counter() - start)
df["ES1"] = np.array(df["Second"].apply(embedding1))
print("third ready")
print(perf_counter() - start)
df["ES2"] = np.array(df["Second"].apply(embedding2))
print("fourth ready")
print(perf_counter() - start)
data = df[["EF1", "EF2", "ES1", "ES2", "Metric"]]
# generate
print(data)
print(perf_counter() - start)

data.to_csv(f"../data/embedding_{V}_{L}.csv")
