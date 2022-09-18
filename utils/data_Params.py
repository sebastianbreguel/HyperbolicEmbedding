##########################
#####    Data Params
##########################

V = 5  # Number of letters
L = 5  # Large of max length of a sequence
N = 500  # Number of samples
S = "abcdefghijklmnopqrstuvwxyz"[:V]
VOCABULARY = [i for i in S]
URL_DISTANCES = f"data/distances/{V}_{L}.csv"
URL_EMBEDDING = f"data/embeddings/{V}_{L}.csv"
