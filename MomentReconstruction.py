import numpy as np
import matplotlib.pyplot as plt
import scipy.special 
from scipy.special import factorial, gamma
import scipy as sp
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon

from functools import partial
import itertools

def phi(x):
    return np.exp(-(1/(1-x**2)))


def generateMoment(f, x, n):
    return sp.integrate.simpson(f*(x**n), x)

def generate2DMoment(f, x, y, p, q):
    return generateMoment(generateMoment(f, x, p), y, q) 

#def C(k, j):
    #return scipy.special.binom(k,j)

def b(k, theta, f):
    sum=0
    for j in range(k+1):
        sum = sum + C(k, j)*(np.cos(theta))**j * (np.sin(theta))**(k-j)* generate2DMoment(f, j, k-j)
    return sum
def c(j):
    if j%2==1:
        return 0
    def moment_density(x):
        return x**j * np.exp(-1/(1-x**2))
    return (-1)**j * sp.integrate.quad(moment_density, -1, 1)

def A_func(m, n,a1, a2, i, j):
    coeff1 = factorial(i)*factorial(j)
    coeff2 = factorial(m-1-a1-i) * factorial(n-1-a2-j)
    return ((-1)**(i+j))/(coeff1*coeff2)


def reconstruct_from_moments(f, moments, m, n, x1_ind, x2_ind):
    x1, x2 = x1_ind/f.shape[0], x2_ind/f.shape[1]
    sum = 0
    a1, a2 = int(np.floor((m-1)*x1)), int(np.floor((n-1)*x2))
    Gamma_sub = moments[a1:,a2:]
    A = np.fromfunction(partial(A_func, m, n , a1, a2), (m-a1, n-a2))
    #print(f"{x1_ind}, {x2_ind}:  (m,n)= ({m},{n}), (a1, a2) = ({a1}, {a2}), Moments Shape = {moments.shape}, gamma shape = {Gamma_sub.shape}")
    '''
    for a1 in range(int(m-np.floor(m*x1))):
        for a2 in range(int(n-np.floor(n*x2))):
            x = (-1)**(a1+a2)/(factorial(a1)*factorial(a2))
            #x = x*generate2DMoment(f, a1+np.floor(m*x1), a2+np.floor(n*x2))
            x = x*moments[int(a1+np.floor(m*x1)), int(a2+np.floor(n*x2))]
            x = x/(factorial(m-np.floor(m*x1)-a1) * factorial(n-np.floor(n*x2)-a2))
            sum = sum + x
      '''
    reconstruction = C(m,n,a1,a2)*np.sum(A*Gamma_sub)
    #print(100*'-'
    #print(f'({x1},{x2})')
    #print(reconstruction, A, Gamma_sub)
    return reconstruction if np.abs(reconstruction)<2 else 0 
    
def C(m,n,a1,a2):
    num = gamma(m+1)*gamma(n+1)
    den = gamma(a1+1)*gamma(a2+1)
    return num/den
    #return (factorial(m+1)*factorial(n+1))/(factorial(np.floor(m*x1))*factorial(np.floor(n*x2)))
    


f = shepp_logan_phantom()   

#x=np.linspace(0,1,400, dtype=np.longdouble)
#y=np.linspace(0,1,400, dtype=np.longdouble)
x=np.linspace(0,1,400, dtype=np.float128)
y=np.linspace(0,1,400, dtype=np.float128)
xx, yy = np.meshgrid(x,y)

f=xx*yy
moments_function = lambda p, q: 1/((p+2)*(q+2))
num = 25

#moments = np.fromfunction(np.vectorize(partial((generate2DMoment), f, x, y)), shape = (num, num))
moments_test = np.fromfunction((moments_function), shape = (num, num), dtype=np.float128)
moments = moments_test
for (i, j) in itertools.product(range(num), range(num)):
        print(f"Moment ({i}, {j}) = {moments_function(i,j) - moments[i,j]}")
print(np.allclose(moments, moments_test))

#def moment_reconstruction(f, moments, num):
#    f_mom = np.zeros(f.shape)
#    for i in range(f_mom.shape[0]):
#        for j in range(f_mom.shape[1]):
#                f_mom[i,j] = reconstruct_from_moments(f, moments, num, num, i, j)
#    return f_mom

#f_mom = moment_reconstruction(f, moments, num)

moment_reconstruction = np.vectorize(partial(reconstruct_from_moments, f, moments, num, num))

#print(moments)
f_mom = np.fromfunction(np.vectorize(partial(reconstruct_from_moments, f, moments, num, num)), shape=f.shape)
print(np.max(f_mom), np.min(f_mom))
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(f_mom/np.max(f_mom), cmap=plt.cm.Greys_r)
ax2.imshow(f, cmap=plt.cm.Greys_r)
plt.show()

print(max(f.shape))
n = max(f.shape)
n =  50
error_fourier = []
error_moments = []
#moments = np.fromfunction(partial(generate2DMoment, f), shape = (n, n))
for i in range(1, n):
    print(i)
    theta = np.linspace(0., 180., i, endpoint=False)
    appf = iradon(radon(f, theta=theta, circle=False), theta=theta, circle=False)
    momf = moment_reconstruction(i, i)
    error_fourier.append(np.sqrt(np.mean((appf-f)**2)))
    error_moments.append(np.sqrt(np.mean((momf-f)**2)))
x = np.array(list(range(1,n)))
plt.plot(error_fourier, label='fourier')
plt.plot(error_moments, label='moments')

plt.plot(x, 2.5/x, "o-", label='y=2.5/x')
plt.legend()
plt.show()




#imkwargs = dict(vmin=-0.2, vmax=0.2

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
#                               sharex=True, sharey=True)
#ax1.set_title("Reconstruction\nFiltered back projection")
#ax1.imshow(appf, cmap=plt.cm.Greys_r)
#ax2.set_title("Reconstruction error\nFiltered back projection")
#ax2.imshow(appf - f, cmap=plt.cm.Greys_r, **imkwargs)
#plt.show()
