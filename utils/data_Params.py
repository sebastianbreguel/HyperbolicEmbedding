##########################
#####    Data Params
##########################
import numpy.random as random

# GANEA

NG = 30000  # Number of examples
R = 0.5  # Prefix Positive Samples
NUM = 9  # Number of OPTIONS
NUMBERS = [str(i) for i in range(0, NUM + 1)]
URL_PREFIX_50 = "data/Prefix/Prefix_50.csv"
URL_PREFIX_30 = "data/Prefix/Prefix_30.csv"
URL_PREFIX_10 = "data/Prefix/Prefix_10.csv"
LARGE = 1
K = 3 * LARGE  # Nuof changes


# MIRCEA

URL = "data/Phylogenetics.csv"
V = 20
NM = 1000
VOCABULARY = "abcdefghijklmnopqrstuvwxyz"[:V]
VOCABULARY = [x for x in VOCABULARY]
MIN_RANDOM = 0
MAX_RANDOM = 0.3
ROUND = 5
# p = [p1, p2, p3, p4, p5, p6]
EMB = {}
for i, e in enumerate(VOCABULARY):
    EMB[e] = i
