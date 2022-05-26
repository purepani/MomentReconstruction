import numpy as np
import scipy as sp
import scipy.sparse.linalg as linalg
from scipy.sparse import lil_array, csc_array, csc_matrix
import einops as eo
from collections import defaultdict
from itertools import zip_longest
import sparse
import time


def getitem(x, index):
    """
    This function implements the indexing functionality for COO.
    The overall algorithm has three steps:
    1. Normalize the index to canonical form. Function: normalize_index
    2. Get the mask, which is a list of integers corresponding to
       the indices in coords/data for the output data. Function: _mask
    3. Transform the coordinates to what they will be in the output.
    Parameters
    ----------
    x : COO
        The array to apply the indexing operation on.
    index : {tuple, str}
        The index into the array.
    """
    from .core import COO

    # If string, this is an index into an np.void

    # Custom dtype.
    if isinstance(index, str):
        data = x.data[index]
        idx = np.where(data)
        data = data[idx].flatten()
        coords = list(x.coords[:, idx[0]])
        coords.extend(idx[1:])

        fill_value_idx = np.asarray(x.fill_value[index]).flatten()
        fill_value = (
            fill_value_idx[0] if fill_value_idx.size else _zero_of_dtype(data.dtype)[()]
        )

        if not equivalent(fill_value, fill_value_idx).all():
            raise ValueError("Fill-values in the array are inconsistent.")

        return COO(
            coords,
            data,
            shape=x.shape + x.data.dtype[index].shape,
            has_duplicates=False,
            sorted=True,
            fill_value=fill_value,
        )

    # Otherwise, convert into a tuple.
    if not isinstance(index, tuple):
        index = (index,)

    # Check if the last index is an ellipsis.
    last_ellipsis = len(index) > 0 and index[-1] is Ellipsis

    # Normalize the index into canonical form.
    index = normalize_index(index, x.shape)

    # zip_longest so things like x[..., None] are picked up.
    if len(index) != 0 and all(
        isinstance(ind, slice) and ind == slice(0, dim, 1)
        for ind, dim in zip_longest(index, x.shape)
    ):
        return x

    # Get the mask
    mask, adv_idx = _mask(x.coords, index, x.shape)

    # Get the length of the mask
    if isinstance(mask, slice):
        n = len(range(mask.start, mask.stop, mask.step))
    else:
        n = len(mask)

    coords = []
    shape = []
    i = 0

    sorted = adv_idx is None or adv_idx.pos == 0
    adv_idx_added = False
    for ind in index:
        # Nothing is added to shape or coords if the index is an integer.
        if isinstance(ind, Integral):
            i += 1
            continue
        # Add to the shape and transform the coords in the case of a slice.
        elif isinstance(ind, slice):
            shape.append(len(range(ind.start, ind.stop, ind.step)))
            coords.append((x.coords[i, mask] - ind.start) // ind.step)
            i += 1
            if ind.step < 0:
                sorted = False
        # Add the index and shape for the advanced index.
        elif isinstance(ind, np.ndarray):
            if not adv_idx_added:
                shape.append(adv_idx.length)
                coords.append(adv_idx.idx)
                adv_idx_added = True
            i += 1
        # Add a dimension for None.
        elif ind is None:
            coords.append(np.zeros(n, dtype=np.intp))
            shape.append(1)

    # Join all the transformed coords.
    if coords:
        coords = np.stack(coords, axis=0)
    else:
        # If index result is a scalar, return a 0-d COO or
        # a scalar depending on whether the last index is an ellipsis.
        if last_ellipsis:
            coords = np.empty((0, n), dtype=np.uint8)
        else:
            if n != 0:
                return x.data[mask][0]
            else:
                return x.fill_value

    shape = tuple(shape)
    data = x.data[mask]

    return COO(
        coords,
        data,
        shape=shape,
        has_duplicates=False,
        sorted=sorted,
        fill_value=x.fill_value,
    )


class BlockToeplitzBlock:
    def __init__(self, N, data):
        self.N=N
        self.data = data




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

