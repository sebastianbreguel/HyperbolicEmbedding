import numpy as np

# Hyperbolic Distance Poincare Model
def poincare_dist(x, y):
    z = 2 * np.linalg.norm(x - y) ** 2
    uu = 1.0 + z / ((1 - np.linalg.norm(x) ** 2) * (1 - np.linalg.norm(y) ** 2))
    return np.arccosh(uu)


# Gyrovectorspace addition of vectors
def mob_add(x, y, c=1):
    numerator = (1.0 + 2.0 * c * np.dot(x, y) + c * np.linalg.norm(y) ** 2) * x + (
        1.0 - c * np.linalg.norm(x) ** 2
    ) * y
    denominator = (
        1.0
        + 2.0 * c * np.dot(x, y)
        + c**2 * np.linalg.norm(y) ** 2 * np.linalg.norm(x) ** 2
    )
    return numerator / denominator


# lambda x
def lambda_x(x, c=1):
    return 2.0 / (1 - c * np.linalg.norm(x) ** 2)


# Gyrovectorspace multiplication of vectors and scalars
def mob_mult(x, r, c=1):
    first = 1 / np.sqrt(c)
    second = np.dot(r, np.arctanh(np.dot(np.sqrt(c), np.linalg.norm(x))))
    third = x / np.linalg.norm(x)
    return first * np.exp(second) * third


# Exponential map
def exp_map_x(x, v, c=1):
    first_term = x
    second_term = np.sqrt(c) * lambda_x(x, c) * np.linalg.norm(v) / 2
    third_term = (np.sqrt(c) * np.linalg.norm(v)) * v
    return mob_add(first_term, np.tanh(second_term * third_term), c)


# log map
def log_map_x(x, y, c=1):
    diff = mob_add(-x, y, c)
    lam = lambda_x(x, c)
    first = 2.0 / (np.sqrt(c) * lam)
    second = np.arctanh(np.sqrt(c) * np.linalg.norm(diff))
    third = diff / np.linalg.norm(diff)
    return first * second * third


# Parallel Trnasport
def parallel_transport(x, v, c=1):
    lambda_0 = lambda_x(0, c)
    lam_x = lambda_x(x, c)
    return np.dot(lambda_0, v) / lam_x


# Euclidean distan
