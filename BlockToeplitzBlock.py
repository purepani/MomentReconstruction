import numpy as np
import scipy as sp
import scipy.sparse.linalg as linalg
from scipy.sparse import lil_array, csc_array, csc_matrix
import einops as eo
from collections import defaultdict
from itertools import zip_longest
import sparse
import time


class Mojette(linalg.LinearOperator):

    def get_Bin_Size(self):
        angle_arr = np.array(self.angles)
        Pmin = np.min(angle_arr[:,0])
        Pmax = np.max(angle_arr[:,0])

        Qmin = np.min(angle_arr[:,1])
        Qmax = np.max(angle_arr[:,1])

        Kmax = self.image_shape[0]
        Lmax = self.image_shape[1]
        
        B_min = (Pmin<0)*Pmin*Lmax-(Qmax>0)*Qmax*Kmax
        B_max = (Pmax>0)*Pmax*Lmax-(Qmin<0)*Qmin*Kmax


        bin_dim_size = B_max - B_min
        return bin_dim_size, B_max, B_min
    

    def __init__(self, N, angles, dtype=None):
        self.angles = angles
        self.image_shape = (N, N)
        self.shape = (self.get_Bin_Size()[0]*len(angles), N**2) 
        self.bin_size, self.bin_max, self.bin_min = self.get_Bin_Size() 
        self.dtype=np.dtype(dtype)
        self.masks_disjoint_bm = self.calculate_disjoint_angles_bm()
        self.masks_disjoint_kl = self.calculate_disjoint_angles_kl()
        
    def calculate_disjoint_angles_kl(self):
        N = self.image_shape[1]
        angles_arr = np.array(self.angles)
        p, q = angles_arr[:, 0], angles_arr[:, 1]

        k, l = np.indices(self.image_shape)

        k = eo.repeat(k, "k l -> (k l m)", m=len(self.angles))
        l = eo.repeat(l, "k l -> (k l m)", m=len(self.angles))

        m, = np.indices(p.shape)
        p = eo.repeat(p, "m -> (k l m)", k=N, l=N)
        q = eo.repeat(q, "m -> (k l m)", k=N, l=N) 
        m = eo.repeat(m, "m -> (k l m)", k=N, l=N)

        masks = list(zip(list(p*l-k*q-self.bin_min), list(m), list(k), list(l)))

        buckets = defaultdict(list)

        for tup in masks:
            buckets[f"{tup[2],tup[3]}"].append(tup)

        masks_separated = list(zip_longest(*buckets.values()))
        masks_separated = [np.array(list(filter(None, mask))) for mask in masks_separated]
        return masks_separated

    def calculate_disjoint_angles_bm(self):
        N = self.image_shape[1]
        angles_arr = np.array(self.angles)
        p, q = angles_arr[:, 0], angles_arr[:, 1]

        k, l = np.indices(self.image_shape)

        k = eo.repeat(k, "k l -> (k l m)", m=len(self.angles))
        l = eo.repeat(l, "k l -> (k l m)", m=len(self.angles))

        m, = np.indices(p.shape)
        p = eo.repeat(p, "m -> (k l m)", k=N, l=N)
        q = eo.repeat(q, "m -> (k l m)", k=N, l=N) 
        m = eo.repeat(m, "m -> (k l m)", k=N, l=N)

        masks = list(zip(list(p*l-k*q-self.bin_min), list(m), list(k), list(l)))

        buckets = defaultdict(list)

        for tup in masks:
            buckets[f"{tup[0],tup[1]}"].append(tup)

        masks_separated = list(zip_longest(*buckets.values()))
        masks_separated = [np.array(list(filter(None, mask))) for mask in masks_separated]
        return masks_separated


    def _matvec(self, image_vector):
        N = self.image_shape[1]
        angles_arr = np.array(self.angles)
        p, q = angles_arr[:, 0], angles_arr[:, 1]

        image = sparse.DOK(eo.rearrange(image_vector, "(l k) -> k l", k = self.image_shape[0]))
        Mojette_image2 = np.zeros((self.bin_size, len(self.angles)))
        #Mojette_image = lil_array((self.bin_size, len(self.angles)))
        Mojette_image = sparse.DOK((self.bin_size, len(self.angles)))

        k, l = np.indices(image.shape)


        k = eo.repeat(k, "k l -> (k l m)", m=len(self.angles))
        l = eo.repeat(l, "k l -> (k l m)", m=len(self.angles))

        m, = np.indices(p.shape)
        p = eo.repeat(p, "m -> (k l m)", k=N, l=N)
        q = eo.repeat(q, "m -> (k l m)", k=N, l=N) 
        m = eo.repeat(m, "m -> (k l m)", k=N, l=N)

        
        #np.add.at(Mojette_image2, (p*l-k*q-self.bin_min, m),  image[k,l])
        #print(np.array([p*l-k*q, m, k, l]).T)

        masks_separated = self.masks_disjoint_bm

        for mask in masks_separated:
            b, m, k, l = (mask[:, i] for i in range(4))
            Mojette_image[b, m] += image[k, l]

        Mojette_image2 = eo.rearrange(Mojette_image2, "b m -> (b m)")
        Mojette_image = eo.rearrange(Mojette_image, "b m -> (b m)")
        print(np.allclose(Mojette_image, Mojette_image2))

        return Mojette_image

    def _rmatvec(self, Mojette_vector):
        N = self.image_shape[1]
        angles_arr = np.array(self.angles)
        p, q = angles_arr[:, 0], angles_arr[:, 1]

        Mojette_image = eo.rearrange(Mojette_vector, "(b m) -> b m", m = len(self.angles))

        back_proj = np.zeros((N, N))

        k, l = np.indices(back_proj.shape)


        k = eo.repeat(k, "k l -> k l m", m=len(self.angles))
        l = eo.repeat(l, "k l -> k l m", m=len(self.angles))

        m, = np.indices(p.shape)
        p = eo.repeat(p, "m -> k l m", k=N, l=N)
        q = eo.repeat(q, "m -> k l m", k=N, l=N) 
        m = eo.repeat(m, "m -> k l m", k=N, l=N)

        #np.add.at(back_proj, (k, l),  Mojette_image[p*l-q*k-self.bin_min, m])

        masks_separated = self.masks_disjoint_kl

        for mask in masks_separated:
            b, m, k, l = (mask[:, i] for i in range(4))
            back_proj[k, l] += Mojette_image[b, m]

        
        back_proj = eo.rearrange(back_proj, "k l -> (l k)")


        return back_proj
        

if __name__ == "__main__":
    N = 3
    angles = [[1, 1], [1,2]]
    M = Mojette(N, angles)
    image = eo.rearrange(np.arange(N**2), "(k l) -> k l", k=N)
    print(image)
    image_vec = eo.rearrange(image, "k l -> (l k)")
    print(image_vec.shape)
    print(eo.rearrange(M*image_vec, "(b m) -> b m", m=len(angles)))

