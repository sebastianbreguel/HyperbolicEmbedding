##########################
#####    Data Params
##########################
import numpy.random as random
# GANEA

NG = 5000  # Number of examples
R = 0.5  # Prefix Positive Samples
NUM = 3 # Number of OPTIONS
NUMBERS = [str(i) for i in range(1, NUM + 1)]
URL_PREFIX_50 = "data/Prefix/Prefix_50.csv"
URL_PREFIX_30 = "data/Prefix/Prefix_30.csv"
URL_PREFIX_10 = "data/Prefix/Prefix_10.csv"
LARGE = 1
K = 3 * LARGE  # Number of changes


# MIRCEA

URL = "data/Phylogenetics.csv"
V = 20
NM = 10000
VOCABULARY = 'abcdefghijklmnopqrstuvwxyz'[:V]
VOCABULARY = [x for x in VOCABULARY]
MIN_RANDOM = 0
MAX_RANDOM = 0.3
p1 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), 2)
p2 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), 2)
p3 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), 2)
p4 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), 2)
p5 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), 2)
p6 = round(random.uniform(low=MIN_RANDOM, high=MAX_RANDOM), 2)
p = [p1, p2, p3, p4, p5, p6]
EMB = {}
for i, e in enumerate(VOCABULARY):
    EMB[e] = i