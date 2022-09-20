##########################
#####    Data Params
##########################

V = 5  # Number of letters
L = 7  # Large of max length of a sequence
N = 1500  # Number of samples
S = "abcdefghijklmnopqrstuvwxyz"[:V]
VOCABULARY = [i for i in S]
K = 3
NG = 50000
R = 0.5
NUM = 10
NUMBERS = [str(i) for i in range(1, NUM + 1)]
URL_DISTANCES = f"data/distances/{V}_{L}.csv"
URL_EMBEDDING = f"data/embeddings/{V}_{L}.csv"
URL_GANEA_15 = "data/Prefix_15/ganea.csv"
URL_GANEA_10 = "data/Prefix_10/ganea.csv"
URL_GANEA_05 = "data/Prefix_05/ganea.csv"
