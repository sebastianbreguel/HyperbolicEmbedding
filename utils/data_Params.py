##########################
#####    Data Params
##########################

V = 5  # Number of letters
L = 7  # Large of max length of a sequence
N = 1500  # Number of samples
S = "abcdefghijklmnopqrstuvwxyz"[:V]
VOCABULARY = [i for i in S]
K = 3
NG = 1000
R = .5
NUM = 3
NUMBERS = [str(i) for i in range(1, NUM + 1)]
URL_DISTANCES = f"data/distances/{V}_{L}.csv"
URL_EMBEDDING = f"data/embeddings/{V}_{L}.csv"
URL_GANEA = "ganea.csv"
