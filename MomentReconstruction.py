import numpy as np
import matplotlib.pyplot as plt
import scipy.special

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon

def xy(X,Y):
    return X*Y

def phi(x):
    return np.exp(-(1/(1-x**2)))
def generateMoment(f, n):
    sum = 0
    for i in range(f.shape):
        sum = sum + f[i]*(i/(f.shape+1))**n

def generate2DMoment(f, n1, n2):
    sum = 0 
    sum = generateMoment(generateMoment(f, n1),n2)
    return sum

def C(k, j):
    return scipy.special.binom(k,j)


def b(k, theta, f):
    sum=0
    for j in range(k+1):
        sum = sum + C(k, j)*(np.cos(theta))**j * (np.sin(theta))**(k-j)* generate2DMoment(f, j, k-j)
    return sum
def c(j):
    




f = shepp_logan_phantom()   


print(max(f.shape))
n = max(f.shape)
n =  100 
for i in range(1, n):
    print(i)
    theta = np.linspace(0., 180., i, endpoint=False)
    appf = iradon(radon(f, theta=theta), theta=theta)
    error.append(np.sqrt(np.mean((appf-f)**2)))
x = np.array(list(range(1,n)))
plt.plot(error)
#plt.plot(x, 2.5/x, "o-"plt.show()



#imkwargs = dict(vmin=-0.2, vmax=0.2

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
#                               sharex=True, sharey=True)
#ax1.set_title("Reconstruction\nFiltered back projection")
#ax1.imshow(appf, cmap=plt.cm.Greys_r)
#ax2.set_title("Reconstruction error\nFiltered back projection")
#ax2.imshow(appf - f, cmap=plt.cm.Greys_r, **imkwargs)
#plt.show()
