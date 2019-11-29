#%%
import numpy as np
import utils.splx_projection.splx_projection as splx

#%%
def admm(M, Y, mu, size):
    Ak = np.zeros(size) # or Y
    Uk = M @ Ak
    Vk = Ak
    Luk = np.zeros(Uk.shape) #1*Uk  # dual
    Lvk = np.zeros(Vk.shape) #1*Vk

    r, imA, F = [], [], []
    I = np.eye(M.shape[1])
    Inv = np.linalg.inv(I + M.T @ M)
    imA.append(Ak)

    for k in range(100):
        Ak =  Inv @ (M.T @ (Uk - Luk) + (Vk - Lvk))
        Nu = (M @ Ak) + Luk
        Uk = 0.5 * (Nu - (1/mu) + np.sqrt((Nu - 1/mu)**2 + (4*Y/mu))) # to be checked again 
        Vk = splx.splx_projection(Ak + Lvk, r=1)
        # Vk = Ak - Lvk
        # Vk[Vk < 0] = 0
        Luk = Luk + (M@Ak - Uk)
        Lvk = Lvk + (Ak - Vk)
        
        imA.append(Ak)
        r.append(np.linalg.norm(M@Ak - Uk, 2))
        # F.append(objective(M, Ak, Y))
    return r, np.array(imA), F