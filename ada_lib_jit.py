import numpy as np
from numba import njit

def func_seed_set(i_seed):
    """
    :param i_seed: integer seed that can be used to set the seed of the functions
    :return: nothing
    """
    np.random.seed(i_seed)

@njit( cache = True, nogil = False )
def seed_generate(N):
    """
    :param N: the number of seeds that you want to generate
    :return: a random selection of ints generate
    """
    return np.random.randint(0,1e9, size = N)

@njit( cache = True, nogil = False )
def func_rand_uniform(a, b, i_seed, N):
    """
    :param A: lower limit of uniform distribution
    :param B: upper limit of uniform distribution
    :param i_seed: seed integer that is used
    :param N: the number of random numbers that we want to generate
    :return: random number from the uniform distribution
    """
    # return (b-a)*np.random.random() + a
    # np.random.seed(i_seed)
    return (b-a)*np.random.random(N) + a


@njit( cache = True, nogil = False )
def func_range(data):
    """
    :param data: input a numpy data array
    :return: returns the range of the data values provided
    """
    return np.max(data) - np.min(data)

@njit( cache = True, nogil = False )
def func_median(data):
    """
    :param data: numpy array containing the data to work with
    :return: the median as either the middle value or the half point between the closest values
    """
    sorted_data = np.sort(data)
    if len(sorted_data)%2 == 0:
        return (sorted_data[int(len(sorted_data)/2)] + sorted_data[int(len(sorted_data)/2-1)])/2
    else:
        return sorted_data[int(len(sorted_data)/2)]

@njit( cache = True, nogil = False )
def func_mean_absolute_deviation(data):
    """
    :param data: numpy array containing the data
    :return: the mean absolute deviation compared to the median of the dataset
    """
    median = func_median(data)
    total = 0
    for i in range(0, len(data)):
        total += np.abs(data[i]-median)

    return 1/len(data) * total

@njit( cache = True, nogil = False )
def func_med_mad(data):
    """
    :param data: numpy array of data
    :return: just returning a tuple of the values
    """
    return func_median(data), func_mean_absolute_deviation(data)

@njit( cache = True, nogil = False )
def func_histogram(data, bins, density):
    """
    :param data: input a numpy data array
    :return: return an array containing the cumulative distribution function
    """
    
    diff = np.max(data) - np.min(data)
    step = diff/bins
    
    hist = np.zeros(bins)

    mins = np.min(data)
    bin_nums = np.zeros(bins)

    old = np.min(data)    
    for i in range(0, bins):
        for j in data:
            if ( j >= old ) and ( j <= old + step ):
                hist[i] += 1
        old +=step
    if len(hist) != len(data):
        hist[-1] += 1

    for bin in range(0, len(bin_nums)):
        bin_nums[bin] =  mins + (bin+0.5)*step

    if density == True:
        hist = hist/(len(data) * step)

    return hist, (hist*step).cumsum(), bin_nums


@njit(cache=True, nogil = False )
def func_rand_gaussian(avg, sigma, seed ,N):
    """
    vectorised function to generate N gaussians numbers quickly and return them.

    :param avg: the average value of the Gaussians generated
    :param sigma: the standard deviation of the generated Gaussians
    :param seed: the seed that you want to seed the generator with
    :param N: the number of Gaussian numbers you would like to generate
    :return: an array of the sampled Gaussian numbers
    """

    x = func_rand_uniform(-1, 1, seed, (N, 2))
    r2 = x[:, 0] ** 2 + x[:, 1] ** 2
    r = np.sqrt(r2)

    empty = N
    output = np.zeros(N)

    while empty != 0:
        for i in range(0, len(r2)):
            if r2[i] < 1 and r[i] > 0:
                if empty != 0:
                    g1 = __calculate_gaussian_val__(x[i][0], r[i])
                    output[N - empty] = sigma*g1+avg
                    empty -= 1
                if empty != 0:
                    g2 = __calculate_gaussian_val__(x[i][1], r[i])
                    output[N - empty] = sigma*g2+avg
                    empty -= 1
        if empty != 0:
            x = func_rand_uniform(-1, 1, seed, (N + empty, 2))[N:]
            r2 = x[:,0] ** 2 + x[:,1] ** 2
            r = np.sqrt(r2)
    return output

@njit( cache = True, nogil = False )
def __calculate_gaussian_val__(x, r):
    """
    :param x: the value of x that will be used in the box-muller transform
    :param r: again a value in the b-m transform
    :return: returns the uniform G(0,1) distribution
    """
    return (2*x/r)*np.sqrt(-np.log(r))

@njit( cache = True, nogil = False )
def func_rand_exponential(tau, i_seed, N):
    """
    :param tau: the exponential decay constant
    :param i_seed: the random seed being used to seed the distribution
    :return: the exponentially generated random number
    """
    u_val = func_rand_uniform(-1,1,i_seed, N)
    return -tau*np.log(1-u_val)

@njit( cache = True, nogil = False )
def func_mean_var(data):
    """
    :param data: numpy array of data
    :return: avg and variance of the data
    """
    avg = 0
    var = 0
    N = len(data)

    for i in range(0,N):
        avg += data[i]
    avg = avg/N

    for i in range(0,N):
        var += (data[i]-avg)**2
    var = var/(N-1)
    return avg, var

@njit( cache = True, nogil = False )
def func_moments(data, m):
    """
    :param data: numpy array of data
    :param m: the m-th moment that we want to calculate the value up to
    :return: a numpy array containing the higher moments of the distribution
    """
    mean, var = func_mean_var(data)
    sigma = np.sqrt(var)

    moments = np.zeros(m)
    N = len(data)
    for i in range(0,m):
        moments [i] = 1/N*np.sum(((((data-mean)/(sigma)))**(i+1)))
    return moments

@njit( cache = True, nogil = False )
def func_invert_mat(matrix):
    """
    :param matrix: a 2x2 matrix that needs to be inverted
    :return: the inverse of the matrix
    """
    mat = np.zeros((2,2))
    a = matrix[0,0]
    b = matrix[0,1]
    c = matrix[1,0]
    d = matrix[1,1]
    det = a*d - b*c

    mat[0,0] = d
    mat[0,1] = -b
    mat[1,0] = -c
    mat[1,1] = a
    return 1/det*mat

@njit( cache = True, nogil = False )
def func_determinant(matrix):
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]
    return a * d - b * c

@njit( cache = True, nogil = False )
def func_chi2(data, fit, var):
    return np.sum(((data-fit)/var)**2)