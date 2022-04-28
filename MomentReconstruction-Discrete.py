import numpy as np
import matplotlib.pyplot as plt
import scipy.special 
from scipy.special import factorial, gamma, comb
import scipy as sp
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon

import functools

def poch(x,q):
    prod = 1
    for i in range(q):
        prod=prod*(x+i)
    return prod

def poch_arr(x,p):
    poch = []
    prod = 1
    poch.append(prod)
    for i in range(p):
        prod=prod*(x+i)
        poch.append(prod)
    return np.array(poch)


def beta(N, p):
    return np.sqrt(factorial(2*p)* comb(N+p, 2*p+1))


def c_func(N, p,r):
    a = (-1)**r / beta(N, p)
    b = factorial(p+r)/(factorial(p-r)*(factorial(r))**2)
    c = poch(1-N, p)/poch(1-N, r)
    return a*b*c

def d(l, q):
    num = (-1)**q * beta(N, q) * (2*q+1)*(factorial(l))**2 * poch(1-N, l)
    den = factorial(l+q+q) * factorial(l-q) * poch(1-N, q)
    return num/den

#def t(N, p, x):
#    sum = 0
#    for r in range(p+1):
#        sum = sum+c(N, p,r)* poch(-x, r)
#    return sum


#def H(N_th, R, p, theta):
#    sum = 0
#    for k in range(N_th):
#        sum = sum + t(N, p,k) * R(theta,k)
#    return sum
    


f = shepp_logan_phantom()

trans = radon(f)
N = f.shape[0]

poch_arr2 = np.fromfunction(poch_arr, poch_shape)

c = np.fromfunction(functools.partial(c_func, N), c_shape)
H = np.einsum("pk, tk->pt", t, trans)    
t = np.einsum("pq, kq -> pk", c, poch_arr2)


plt.imshow(trans)
plt.show()
