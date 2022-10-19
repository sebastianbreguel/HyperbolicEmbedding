from .train_functions import obtain_loss, obtain_optimizer, train_model, val_process
from .model_data import get_data, get_model, getMNIST
from .parameters import EPOCHS, LEARNING_RATE, SEED, DIMENTIONS, USE_BIAS
from .stadistic_util import get_accuracy, get_metrics
from .run import run_model, run_MNIST
from .gen_data import generate_data
