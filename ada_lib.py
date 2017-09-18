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

def func_range(data):
    """
    :param data: input a numpy data array
    :return: returns the range of the data values provided
    """
    return np.max(data) - np.min(data)

def func_cum_dist(data):
    """
    :param data: input a numpy data array
    :return: return an array containing the cumulative distribution function
    """
    sorted_data = np.sort(data)
    length = len(data)
    total = np.zeros(length)
    for i in range(0, length):
        total[i] += sorted_data[i]
    return total

def func_rand_gaussian(avg, sigma, i_seed):
    """
    :param avg: average value of the gaussian distribution that we want to create
    :param sigma: the standard deviation of the distribution
    :param i_seed: an array of 2 seed values that we will use when generating our values
    :return: an array containing 2 values scaled appropriately for each distribution
    """
    x_val = func_rand_uniform(-1,1,i_seed[0])
    y_val = func_rand_uniform(-1,1,i_seed[1])

    r2 = x_val**2 + y_val**2
    r = np.sqrt(r2)

    if r2 < 1 and r > 0:
        g1 = __calculate_gaussian_val__(x_val, r)
        g2 = __calculate_gaussian_val__(y_val, r)
        return np.array([sigma*g1+avg, sigma*g2+avg])

    else:
        return func_rand_gaussian(avg, sigma, np.random.randint(0,1e9, size = 2))

def __calculate_gaussian_val__(x, r):
    """
    :param x: the value of x that will be used in the box-muller transform
    :param r: again a value in the b-m transform
    :return: returns the uniform G(0,1) distribution
    """
    return (2*x/r)*np.sqrt(-np.log(r))

def func_rand_exponential(tau, i_seed):
    """
    :param tau: the exponential decay constant
    :param i_seed: the random seed being used to seed the distribution
    :return: the exponentially generated random number
    """
    u_val = func_rand_uniform(-1,1,i_seed)
    return -tau*np.log(1-u_val)

def func_mean_var(data):
    """
    :param data: numpy array of data
    :return: avg and variance of the data
    """

    # return np.mean(data), np.var(data)

    avg = np.sum(data)/len(data)
    var = 1/len(data)*np.sum((data-avg)**2)
    return avg, var