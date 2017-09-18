import numpy as np

def func_rand_uniform(a, b, i_seed):
    """
    :param A: lower limit of uniform distribution
    :param B: upper limit of uniform distribution
    :param i_seed: seed integer that is used
    :return: random number from the uniform distribution
    """

    np.random.seed(i_seed)
    return (b-a)*np.random.random() + a