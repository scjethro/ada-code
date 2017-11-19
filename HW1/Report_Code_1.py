# 130003186 HW1 Code

# import statements of useful modules
import numpy as np, matplotlib.pyplot as plt
# numpy is the numerical computing library for python, useful for vectorised operations
# matplotlib is a plotting library that interacts with numpy
plt.style.use('bmh')
# set the plot style to something acceptable

import ada_lib_jit as ada
# this library is my current ADA library


from numba import njit
# just in time compiler giving C/Fortran level performance to Python code (very useful!)
# using the njit() version to remove python interactivity and give faster code


from smtplib import SMTP_SSL
# used for the first question

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#																							  # 
# Question 1																				  #
#																							  # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

msg = """To: Keith Horne <kdh1@st-andrews.ac.uk>
Subject: Short Email Message

Dear Keith,

This is a short email message sent using the smtplib module in Python as part of the AS5001 ADA module.

Kind regards,

***
"""


with SMTP("smtp.gmail.com", port = 587) as smtp:
    smtp.starttls()
    smtp.login('***@st-andrews.ac.uk', '***')
    smtp.sendmail('***@st-andrews.ac.uk', '***@st-andrews.ac.uk',msg)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#																							  # 
# Question 2																				  #
#																							  # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# define a set of functions which can generate the data necessary for Q2

i_seed = 100

def hist_plot(N, bins):
	# generate values from a random uniform distribution with seed i_seed
    values = ada.func_rand_uniform(-1,1,i_seed,N)
    # then use my cumulative distribution function and histogram function to bin the data
    cum_dist = ada.func_histogram(values, bins)[1]
    return values, cum_dist

def gauss_plot(N):
    # generate N values from a gaussian data set, these are from the normal distribution at the moment
    values = ada.func_rand_gaussian(0,1,i_seed,N)
    # then use my cumulative distribution function and histogram function to bin the data
    # have set the bin number to 100 by default here just because I found it gave to the nicest graphs
    cum_dist = ada.func_histogram(values, 100)[1]
    return values, cum_dist

def exp_plot(N):
	# generate N values from the exponential distribution
    values = ada.func_rand_exponential(5,i_seed,N)
    # then use my cumulative distribution function and histogram function to bin the data
    # have set the bin number to 100 by default here just because I found it gave to the nicest graphs
    cum_dist = ada.func_histogram(values, 100)[1]
    return values, cum_dist

# I am aware that I could abstract this further and simply define a single function that would do all of this,
# given the similarity of the individual functions however for my own sanity and for debugging I kept them as separate functions.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#																							  # 
# Generating plots	for Q 2																	  #
#																							  # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Uniform Plots

# define the number of points we want, this can obviously be changed
N_uniform = 10000 
n_bins = 101
# call our function to generate the data
data_uniform = hist_plot(N_uniform, n_bins)

# create a plotting space
f, ax = plt.subplots(2)
# generate a linear x space
x = np.linspace(-1,1,num = n_bins)

# call histogram plot to plot the data, give it a linear range for the bins
ax[0].hist(data[0],bins = x)
# then plot the uniform cumulative distribution function
ax[1].plot(x,data_uniform[1])

# we set the properties of the plots
# I am using f-strings to set the titles which allows me to very easily include variables
ax[0].set_xlim(-1,1)
ax[0].set_title(f'Histogram for N = {N_uniform}', ha = 'center')
ax[0].set_xlabel('Data')
ax[0].set_ylabel('Count')

ax[1].set_xlim(-1,1)
ax[1].set_title(f'Cumulative Distribution for N = {N_uniform}', ha = 'center')
ax[1].set_xlabel('Data')
ax[1].set_ylabel('Cumulative Value')

plt.tight_layout()
# we save the figure and use tight_layout to stop any overlaps
# plt.savefig('./HW1/Report/Images/problem2_10000.pdf', dpi = 800, format = 'pdf')
plt.show()


# Gaussian Plot

N_gauss = 10000
data_gauss = gauss_plot(N_gauss)

f, ax = plt.subplots(2)
x = np.linspace(-4,4,num = nbins)
ax[0].hist(data[0], bins = nbins)
ax[1].plot(x, data[1])

ax[0].set_title(f'Histogram for N = {N_gauss}', ha = 'center')
ax[0].set_xlabel('Data')
ax[0].set_ylabel('Count')

ax[1].set_title(f'Cumulative Distribution for N = {N_gauss}', ha = 'center')
ax[1].set_xlabel('Data')
ax[1].set_ylabel('Cumulative Value')

plt.tight_layout()

# plt.savefig('./HW1/Report/Images/problem2_gaussian_10k.pdf', dpi = 800, format = 'pdf')
plt.show()

# Exponential Plot

N_exp = 10000
data = exp_plot(N_exp)

f, ax = plt.subplots(2)
x = np.linspace(0,N_exp,num = nbins)
ax[0].hist(data[0], bins = nbins)
ax[1].plot(x,data[1])

ax[0].set_title(f'Histogram for N = {N_exp}', ha = 'center')
ax[0].set_xlabel('Data')
ax[0].set_ylabel('Count')

ax[1].set_title(f'Cumulative Distribution for N = {N}', ha = 'center')
ax[1].set_xlabel('Data')
ax[1].set_ylabel('Cumulative Value')

plt.tight_layout()

# plt.savefig('./HW1/Report/Images/problem2_exponential_10k.pdf', dpi = 800, format = 'pdf')

plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#																							  # 
# Question 3																				  #
#																							  # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

bias_data = np.genfromtxt('./HW1/bias.dat.csv', delimiter= ',', skip_header=1)
# define all of the values required
x_pix = data[:,0]
y_pix = data[:,1]
bias  = data[:,2]

bias_mean, bias_variance = ada.func_mean_var(bias)
bias_median, bias_mad = ada.func_med_mad(bias)
N = len(bias)

print(f'sample mean {bias_mean}, standard deviation {(4*N-4)/(4*N-5)*np.sqrt(bias_variance)}')
print(f'bias median {bias_median}, bias MAD {bias_mad}')
print(f'the error in the mean is then {np.sqrt(bias_variance/len(bias))}')
var = (2*bias_variance**4)/(N-1)
std_err = (4*N-4)/(4*N-5)*np.sqrt(var)
print(f'the error in the standard deviation is {std_err}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                                             # 
# Question 4                                                                                  #
#                                                                                             # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Uniform Data

uniform_moment_data = ada.func_rand_uniform(-1,1, 100, int(10e6))
uniform_moments = ada.func_moments(uniform_moment_data,4)
print(f'the computed uniform moments are {uniform_moments}')

# Gaussian Data

gaussian_moment_data = ada.func_rand_gaussian(0,1, 100, int(10e6))
gaussian_moments = ada.func_moments(gaussian_moment_data,4)
print(f'the computed gaussian moments are {gaussian_moments}')

# Exponential Data

exponential_moment_data = ada.func_rand_exponential(5, 100, int(10e6))
exponential_moments = ada.func_moments(exponential_moment_data,4)
print(f'the computed exponential moments are {exponential_moments}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                                             # 
# Question 5                                                                                  #
#                                                                                             # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# define a function to perform the iteration that we require to compute the monte-carlo scattering

def func_mc(N, M):
    med = np.zeros(M)
    mean = np.zeros(M)
    seeds = ada.seed_generate(M)
    # generate the correct number of seeds to be passed into the function, can pass only a single one if we want to seed it once

    # loop over the correct number of times and then compute N samples
    for i in range(0,M):
        data = ada.func_rand_gaussian(1,1,seeds[i], N)
        med[i] = ada.func_med_mad(data)[0]
        mean[i] = ada.func_mean_var(data)[0]
    
    mean_mean, var_mean = ada.func_mean_var(mean)
    mean_med, var_med = ada.func_mean_var(med)
    
    return np.array([mean_mean, var_mean, mean_med, var_med])

N = 100
M = 10000
# pre-allocate the memory to keep things fast
monte_carlo_data = np.zeros((N,4))

# loop over using a generic for loop and continually call the func_mc function with different values of N each time
for i in range(1,N+1):
    monte_carlo_data[i-1] = func_mc(i, M)

f, ax = plt.subplots()

ax.loglog(np.linspace(1,101,100),np.sqrt(monte_carlo_data[:,1]))
ax.loglog(np.linspace(1,101,100),np.sqrt(monte_carlo_data[:,3]))

ax.set_title('Problem 5a')
ax.set_xlabel('N Samples')
plt.rc('text', usetex = True)
ax.set_ylabel(r'$\sigma$', fontsize = 20)

ax.legend(['Mean', 'Median'], fontsize = 15)

# plt.savefig('./HW1/Report/Images/problem5_plot.pdf', dpi = 800, format = 'pdf')

plt.show()
