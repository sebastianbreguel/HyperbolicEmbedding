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
#####    Ganea Task
##########################
NGS = 50_000  # number of examples
WORDS = 1_000
POSITIVE = 0.5  # fraction of positive prefix samples
NUM = 9  # number of character options
NUMBERS = [str(i) for i in range(0, NUM + 1)] * 4
URL_PREFIX_50 = "data/Prefix/Prefix_50"
URL_PREFIX_30 = "data/Prefix/Prefix_30"
URL_PREFIX_10 = "data/Prefix/Prefix_10"
LARGE = 2
WORD_LARGE = 20

##########################
#####    Mircea Task
##########################
URL = "data/Phylogenetics.csv"
V = 20
NM = 1000
VOCABULARY = list("abcdefghijklmnopqrstuvwxyz"[:V])
MIN_RANDOM = 0
MAX_RANDOM = 0.3
ROUND = 5
EMB = {e: i for i, e in enumerate(VOCABULARY)}

##########################
#####    MNIST Task
##########################
DIMENSIONS = 15
