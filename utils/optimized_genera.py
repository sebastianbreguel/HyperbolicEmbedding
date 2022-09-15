from functions import *
from parameters import *
import multiprocessing

words = [" "] * N
for _ in range(N):
    a = generate_word(L, VOCABULARY)
    words[_] = a
df_words = pd.DataFrame(words, columns=["word"]).drop_duplicates()
# reset the index of the dataframe


df_words = df_words.reset_index(drop=True)
n = len(df_words)
total = n * n

lista = [[]] * total


def func(index: int, wenlo: int, return_dict: dict):
    for i in range(wenlo * (index - 1), wenlo * index):
        try:
            a = df_words.iloc[i]["word"]
            for j in range(i + 1, n):
                # select two random words and then aply metric

                b = df_words.iloc[j]["word"]
                return_dict[i * n + j] = [a, b, metric(a, b)]
        except:
            pass


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    wenlo = int(n / 8)

    for i in range(1, 9):
        print(i)
        p = multiprocessing.Process(target=func, args=(i, wenlo, return_dict))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    # print(return_dict)
    for i in return_dict:
        try:
            lista[i] = return_dict[i]
        except:
            pass
    # print(lista)
    df_distances = pd.DataFrame(lista, columns=["words 1", "words 2", "metric"])

    df_distances.to_csv(URL)
    df_words.to_csv(f"../data/words____{V}_{L}.csv")
