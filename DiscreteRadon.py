import numpy as np
import scipy as sp
from scipy import integrate as integrate
import scipy.sparse
import sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import einops as eo
import time
from functools import partial

class Farey:
    def __init__(self, N):
        self.N = N
    @classmethod
    def seq_len(cls, N, Fdict = dict()):
        if N==1:
            return 2
        sum = ((N+3)*N)//2
        for i in range(2, N+1):
            ind = N//i
            if not (ind in Fdict):
                Fdict[ind] = cls.seq_len(ind, Fdict)
            sum = sum - Fdict[ind]
        return sum

    def __len__(self):
        return self.seq_len(self.N)


    def __iter__(self):
        self.i = 0
        self.prev_items = []
        return self

    def __next__(self):
        ret = None
        if self.i==0:
            self.prev_items.append((0, 1))
            ret = self.prev_items[0]
        elif self.i==1:
            self.prev_items.append((1, self.N))
            ret =self.prev_items[1]
        elif self.prev_items[-1]==(1,1):
            raise StopIteration
        else:
            a, b = self.prev_items[0]
            c, d = self.prev_items[1]
            k = (self.N+b)//d
            p = k*c-a
            q = k*d-b
            self.prev_items[0] = self.prev_items[1]
            self.prev_items[1] = (p,q) 
            ret = (p,q)
        self.i = self.i+1
        return ret 


def get_Bin_Size(N, angles):
    angle_arr = np.array(angles)
    Pmin = np.min(angle_arr[:,0])
    Pmax = np.max(angle_arr[:,0])

    Qmin = np.min(angle_arr[:,1])
    Qmax = np.max(angle_arr[:,1])

    Kmax = N 
    Lmax = N 
    
    B_min = (Pmin<0)*Pmin*Lmax-(Qmax>0)*Qmax*Kmax
    B_max = (Pmax>0)*Pmax*Lmax-(Qmin<0)*Qmin*Kmax


    bin_dim_size = B_max - B_min
    return bin_dim_size, B_max, B_min

class BlockToeplitzBlock:
    pass

def harr_pixel_model(x, y):
    x, y = (np.asarray(w) for w in [x, y])
    return ((x<0.5)&(y<0.5))*1 + ((x==0.5)&(y==0.5))*0.5

def project_pixel_model(angles, bins, pixel_model, b, m):
    shape = b.shape
    #p = eo.rearrange(angles[:, 0], "p -> k l b p", k=shape[0], l=shape[1], b=shape[2], m=shape[3])
    #q = eo.rearrange(angles[:, 1], "p -> k l b p", k=shape[0], l=shape[1], b=shape[2], m=shape[3])
    p = angles[:, 0]
    q = angles[:, 1]
    def integrand(x, b, p, q):
        if p!=0:
            y = (b+q*x)/p
            return pixel_model(x, y)
        else:
            y = (q*x-b)/q
            return pixel_model(y, x)

    def quad_vec(b, p, q):
        return integrate.quad(integrand, -np.inf, np.inf, args=(b, p, q))

    quadv = np.vectorize(quad_vec)
    return quadv(bins[b], p[m], q[m]) 

def convolve(a, b, axis=0):
    return np.abs(np.fft.ifft(np.fft.fft(a, axis=axis) * np.fft.fft(b, axis=axis), axis=axis))


def mojette_operator_pm(pixel_model, angles, bins, k, l, b, m):
    shape = b.shape
    angles = np.asarray(angles)
    b_length = (np.max(bins)-np.min(bins))/shape[2]
    pmp = np.fromfunction(partial(project_pixel_model, angles, bins, pixel_model), (np.max(b)+1, np.max(m)+1) )
    angles = eo.rearrange(angles[:, 0], "m -> k l b m", k=shape[0], l=shape[1], b=shape[2], m=shape[3])

    p = eo.rearrange(angles[:, 0], "m -> k l b m", k=shape[0], l=shape[1], b=shape[2], m=shape[3])
    q = eo.rearrange(angles[:, 1], "m -> k l b m", k=shape[0], l=shape[1], b=shape[2], m=shape[3])
    return convolve()


def multi_int(integrand, a, b):
    ls = []
    for lower in a:
        for upper in b:
            ls.append(integrate.quad(integrand, lower, upper))
    return np.array(ls).reshape(a.shape)


def dirac_mojette(N, ind, angles, B_min, image, bin_size):
    for k in range(N):
        for l in range(N):
            for m, (p, q) in enumerate(angles):
                ind[0].append(k)
                ind[1].append(l)
                ind[2].append(p*l-q*k-B_min)
                ind[3].append(m)
            if k%5==0 and l%5==0:
                print(f"{(k, l)}")
     
    data = 1
    image_vec = sparse.COO(eo.rearrange(image, "k l -> (l k)"))
    M = sparse.COO(ind, data, shape =(N, N, bin_size, len(angles))).transpose((2, 3, 1, 0)).reshape((bin_size*len(angles), N*N))

    start = time.time()    
    moj_trans = M@image_vec
    end = time.time()
    print(f"Time for transform: {end-start}")

    start = time.time()    
    back_proj = M.T@moj_trans
    end = time.time() 
    print(f"Time for backproj: {end-start}")
    
    
    #MsM = sparse.COO((N*N, N*N)) 

    MsM = M.T@M
    #print("Done constructing MsM")

    MstarMop = splinalg.aslinearoperator(MsM)
    reconstruction, err = splinalg.cg(MstarMop, back_proj.todense(), tol=1e-8, callback= lambda x: print(f"current guess: {x}"))

    reconstruction = eo.rearrange(reconstruction, "(l k) -> k l", k=N)
    fig, ax = plt.subplots(2)
    ax[0].imshow(reconstruction)
    ax[1].imshow(image)
    plt.show()

    print(np.allclose(reconstruction, image, atol=1e-1))


def main():
    N=int(input("Enter image size: "))
    min_katz = np.ceil((-1+np.sqrt(1+2*N))/2)
    print(f"Min F for reconstruction is: {min_katz}")
    F=int(input("Enter Farey index: "))
    image = np.arange(N*N).reshape((N,N))

    angles = list(Farey(F))
    angles = angles + [(-p, q) for p,q in angles]
    
    bin_size, B_max, B_min = get_Bin_Size(N, angles)
    
    ind = [[],[],[],[]]

    pixel_model = harr_pixel_model

    bins = np.linspace(-2,2, 11)
    #angles = np.array([[1, 2]])
    
    test, err = np.fromfunction( partial(project_pixel_model, np.array(angles), bins, pixel_model), (bins.size, len(angles)), dtype=np.int32)
    print(test)




if __name__ == "__main__":
    main()

