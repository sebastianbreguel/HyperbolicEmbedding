##########################
#####    Model Params
##########################
EPS = 1e-3
EPOCHS = 60
BATCH_SIZE = 128 * 4
LEARNING_RATE = 0.0065
IN_FEATURES = 140
OUT_FEATURES = 2
SEED = 0
USE_BIAS = False


##########################
#####    Data Params
##########################

# GANEA

NGS = 50_000  # Number of examples
WORDS = 1_000
POSITIVE = 0.5  # Prefix Positive Samples
NUM = 9  # Number of OPTIONS
NUMBERS = [str(i) for i in range(0, NUM + 1)] * 4
URL_PREFIX_50 = "data/Prefix/Prefix_50"
URL_PREFIX_30 = "data/Prefix/Prefix_30"
URL_PREFIX_10 = "data/Prefix/Prefix_10"
LARGE = 2
WORD_LARGE = 20

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


# MNIST

DIMENTIONS = 15
