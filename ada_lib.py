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

def func_median(data):
    """
    :param data: numpy array containing the data to work with
    :return: the median as either the middle value or the half point between the closest values
    """
    sorted_data = np.sort(data)
    if len(sorted_data)%2 == 0:
        return (sorted_data[int(len(sorted_data)/2)] + sorted_data[int(len(sorted_data)/2-1)])/2
    else:
        return sorted_data[int(len(sorted_data)/2)+1]

def func_mean_absolute_deviation(data):
    """
    :param data: numpy array containing the data
    :return: the mean absolute deviation compared to the median of the dataset
    """
    median = func_median(data)
    return 1/len(data)*np.sum(np.abs(data-median))

def func_med_mad(data):
    """
    :param data: numpy array of data
    :return: just returning a tuple of the values
    """
    return func_median(data), func_mean_absolute_deviation(data)

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
    avg = np.sum(data)/len(data)
    var = 1/len(data)*np.sum((data-avg)**2)
    return avg, var

def func_moments(data, m):
    """
    :param data: numpy array of data
    :param m: the m-th moment that we want to calculate the value up to
    :return: a numpy array containing the higher moments of the distribution
    """
    moments = np.zeros(m)
    mean, var = func_mean_var(data)
    N = len(data)
    for i in range(0,m):
        moments[i] = 1/N*np.sum(((data-mean)/np.sqrt(var))**(m+1))
    return moments