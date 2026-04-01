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
#####    Model Architecture
##########################
HIDDEN_SIZE = 64

##########################
#####    MNIST Task
##########################
DIMENSIONS = 15

##########################
#####    Config API
##########################

_OVERRIDABLE = {
    "epochs": "EPOCHS",
    "learning_rate": "LEARNING_RATE",
    "batch_size": "BATCH_SIZE",
    "hidden_size": "HIDDEN_SIZE",
    "use_bias": "USE_BIAS",
    "seed": "SEED",
    "dimensions": "DIMENSIONS",
}

# Snapshot of original defaults at import time — immune to apply_config mutations
_ORIGINAL_DEFAULTS: dict[str, object] = {
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "use_bias": USE_BIAS,
    "seed": SEED,
    "dimensions": DIMENSIONS,
}


def get_defaults() -> dict:
    """Return the original default values (unaffected by apply_config)."""
    return dict(_ORIGINAL_DEFAULTS)

def apply_config(overrides: dict) -> None:
    """Override config module attributes at runtime."""
    import sys
    module = sys.modules[__name__]
    for key, value in overrides.items():
        if key in _OVERRIDABLE:
            setattr(module, _OVERRIDABLE[key], value)
