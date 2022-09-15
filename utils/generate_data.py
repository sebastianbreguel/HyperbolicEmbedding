from functions import *
from parameters import *
from time import perf_counter

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
    if i % 100 == 0:
        end = perf_counter()
        arrows = int(value / total * 10 // 1)
        arrows = "-" + "-" * (arrows) + ">" + " " * (10 - arrows)
        print(f"\n\nLAST: {(end - start):.2f} seconds")
        start = perf_counter()
        print(
            f"{i}/{n}{arrows}{((value/total * 1000)//1)/10}% \n|total time:{((end - initial)/60):.2f} minutes\n"
        )

end = perf_counter()
arrows = "-" + "-" * (10) + ">"
print(f"\n\nLAST: {(end - start):.2f} seconds")
print(
    f"{i}/{n}{arrows}{((value/total * 1000)//1)/10}% \n|total time:{((end - initial)/60):.2f} minutes\n"
)

# # dataframe of the words and the distance
df_distances = pd.DataFrame(lista, columns=["words 1", "words 2", "metric"])

df_distances.to_csv(URL)
df_words.to_csv(f"../data/words____{V}_{L}.csv")
