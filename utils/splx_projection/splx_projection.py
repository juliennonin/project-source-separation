import numpy as np
from sklearn.utils import check_random_state

def splx_projection(y, r):
    """
    splx_projection: Projection onto the r-simplex
    
    Project each column of the matrix y onto the r-simplex.
    
    Parameters
    ----------
    y : array
        Input array.
    r : float
        Radius of the simplex
    
    Returns
    -------
    z: array
        Projection of each column of y onto the r-simplex.
    
    Author: P.-A. Thouvenin
    Original MATLAB code from L. Condat, available at:
    [https://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/download/proj_simplex_l1ball.m]
    """

    z = np.maximum(y - np.max( (np.cumsum(np.sort(y, axis=0)[::-1,:], axis=0) - r) 
                       / np.arange(1, y.shape[0] + 1)[:, np.newaxis], axis=0),
                   0.)
    return z     

if __name__ == "__main__":

    rng = check_random_state(0)
    a = rng.randn(5,5)

    x = splx_projection(a, 1.)

    print("Sum of column:", np.sum(x, axis=0))
    print("All non-negative element?", not np.any(x<0))

    pass