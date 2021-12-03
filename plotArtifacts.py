import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp #from scipy import optimize from functools import partial
from skimage.data import shepp_logan_phantom 
from skimage.transform import radon, rescale, iradon

'''
def broken_detector_sp(x1_num, x2_num, k_num): x1, x2, phi, k = sp.symbols('x1 x2 phi k')
    #expr = x1*(sp.cos(phi))*(sp.cos(phi)) +x2*sp.cos(phi)*sp.sin(phi)-k
    #expr = x1*phi**2 +x2*phi*sp.sqrt(1-phi**2)-k
    expr =x1*sp.cos(phi) + x2*sp.sin(phi) - k
    expr = expr.subs(x1, x1_num).subs(x2, x2_num).subs(k, k_num)
    sol= sp.solveset(expr, phi)
    #sp.pprint(sol.args[0])
    n = sp.symbols("n", integer=True)
    print(type(sp.imageset(sp.cos,sol).args[0].args[0].args[1]))
    test = (sp.trigsimp(sp.imageset(sp.cos, sol).args[0].args[0]))
    print('test')
    print(test)
    return map(sp.N, sol)
    #return (sp.imageset(sp.cos, sol), sp.imageset(sp.sin, sol))
'''
def broken_detector(x1, x2, k, phi):
    return x1*np.cos(phi)+x2*np.sin(phi) - k


def broken_phi(point, k):
    x1=point[0]
    x2=point[1]
    x = x1+1j*x2
    def quad_roots(a, b, c):
        disc = np.sqrt(b**2-4*a*c)
        return np.array([(-b +disc)/(2*a), (-b-disc)/(2*a)])
    roots=quad_roots(np.conj(x), -2*k, x)
    return np.real(np.log(roots)/1j)


def broken_lines(point, k, xx):
    broken_phis = broken_phi(point, k)
    broken_lines = [(-np.cos(phi)/np.sin(phi))*(xx-point[0])+point[1] for phi in broken_phis.flatten()]
    return broken_lines


def broken_sinogram(sinogram, broken_ind):
    sinogram_broken = sinogram[:]
    mask = np.zeros(sinogram.shape)
    mask[broken_ind] = 1
    sinogram_broken[mask==1] = 0 
    return sinogram_broken

def main():
    k=1.1
    point = [10, 10]
    tlim=np.array([0,5])
    tt=np.linspace(tlim[0],tlim[1], 20)

    circle = plt.Circle((0, 0), k, color='b', fill=False)
    plt.style.use('ggplot')
    plt.gca().add_patch(circle)
    plt.axvline(0, color = 'black')
    plt.axhline(0, color = 'black')

    xx=np.linspace(-20,20,1000)
    for line in broken_lines(point, k+tt, xx):
        #plt.plot(xx, line)
        pass
    #plt.xlim([-20,20])
    #plt.ylim([-20,20]) 
    #plt.show()

    grid=plt.GridSpec(2, 2)
    g1 = plt.subplot(grid[0, 0])
    g2 = plt.subplot(grid[0, 1])
    g3 = plt.subplot(grid[1, :])

    f = shepp_logan_phantom()
    theta = np.linspace(0., 360., 1440, endpoint=False)

    sinogram = radon(f, theta=theta)
    print(sinogram.shape)
    recf = iradon(sinogram, theta=theta)
    #sinogram_broken = radon(f, theta=theta)
    broken_top = [150,152]
    broken_bottom = [sinogram.shape[0]-x for x in broken_top]
    print(broken_bottom)
    #indecies = np.r_[broken_top[0]:broken_top[1], broken_bottom[1]:broken_bottom[0]]
    indecies = np.r_[broken_top[0]:broken_top[1]]
    print(indecies)
    sinogram_broken=broken_sinogram(sinogram, (indecies))

    g3.imshow(sinogram_broken, cmap=plt.cm.Greys_r)
    appf = iradon(sinogram_broken, theta=theta) 

    g2.imshow(appf, cmap=plt.cm.Greys_r)
    g1.imshow(recf-appf, cmap=plt.cm.Greys_r)
    #plt.imshow(recf, cmap=plt.cm.Greys_r)
    plt.show()
    #plt.imshow(appf-f, cmap=plt.cm.Greys_r)
    #plt.show()


if __name__== '__main__':
    main()
