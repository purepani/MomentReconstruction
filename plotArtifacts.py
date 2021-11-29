import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
#import scipy.special

#from skimage.data import shepp_logan_phantom
#from skimage.transform import radon, rescale


def broken_detector(x1_num, x2_num, k_num):
    x1, x2, phi, k = sp.symbols('x1 x2 phi k')
    expr = x1*(sp.cos(phi))*(sp.cos(phi)) +x2*sp.cos(phi)*sp.sin(phi)-k
    #expr = x1*phi**2 +x2*phi*sp.sqrt(1-phi**2)-k
    expr = expr.subs(x1, x1_num).subs(x2, x2_num).subs(k, k_num)
    sol= sp.solve(expr, phi)
    #sp.pprint(sol.args[0])
    return sol
    #return (sp.imageset(sp.cos, sol), sp.imageset(sp.sin, sol))

def main():
    r=3
    point = [5,5]
    broken_lines = broken_detector(*point,r)
    #circle = plt.Circle((0, 0), r, color='b', fill=False)
    #plt.gca().add_patch(circle)
    plot = sp.plotting.plot(None)
    sp.pprint(broken_lines)
    for phi in broken_lines:
        x = sp.symbols("x")
        m = (r*(sp.sin(phi))-(point[1]))/(r*(sp.cos(phi))-(point[0]))
        line = m*(x-(point[0]))+point[1]
        sp.pprint(sp.N(sp.cos(phi)*line))
        plot.extend(sp.plotting.plot(line, show=False))
        print(plot)
    #sp.plotting.plot(point)
    plot.show()
if __name__== '__main__':
    main()

