import numpy as np

def chi_square(data_array, model_array, error_array):
    if len(data_array) == len(model_array) == len(error_array):
        return np.sum(((data_array - model_array)/(error_array))**2)

