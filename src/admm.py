#%%
import numpy as np

#%%
def admm(M, Y, mu, size):
    assert np.all(M[0,0] * np.eye(M.shape[0]) == M)
    coef = M[0,0]
    Ak = np.zeros(size)
    Sk = M @ Ak
    Uk = 0*Sk
    r, imA, F = [], [], []
    imA.append(Ak)
    
    for k in range(100):
        Nu = (M @ Ak) - Sk
        Sk = 0.5 * (Nu - (1/mu) + np.sqrt((Nu - 1/mu)**2 + (4*Y/mu)))
        # Ak = splx.splx_projection(Sk - Uk) # Ak - Uk
        Ak = Sk - Uk  # Ak - Uk
        Ak[Ak < 0] = 0
        Ak /= coef
        Uk = Uk + (M @ Ak - Sk)
        imA.append(Ak)
        # F.append(objective(M, Ak, Y))
        r.append(np.linalg.norm(M@Ak - Sk, 2))
    return r, np.array(imA), F