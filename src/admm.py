#%%
import numpy as np
import utils.splx_projection.splx_projection as splx
import src.fbpd as fbpd

from IPython.display import clear_output

#%%
def objective(U, Y):
    # indicatrix of positive values
    I = np.copy(U)
    I[I<=0] = np.inf

    # ylog(u+) with convention 0log(0) = 0
    L = np.maximum(U, 0)
    mask1 = (L == 0)  # if log(0)
    L = np.log(L, where=(L!=0))
    mask2 = (Y != 0)  # if not y == 0
    L = Y * L
    L[mask1 & mask2] = - np.inf # log(0) = - np.inf

    return np.sum(U + I - L)

#%%
def admm(M, Y, rho, alpha, size, max_iter=100):
    """ADMM without regularization parameter
    Blind source separation with Poisson noise
    
    Arguments:
        M {array} -- linear operator modeling the acquisition
        Y {array} -- acquired data with Poisson noise
        rho {positive float} -- penalty paramater
        alpha {positive float} -- regularization parameter
        size {tuple} -- size of the abundacy map to restore

    Returns:
        norms_primal_U {list} -- norm of primal residual for U for each iteration
        objectives {list} -- evaluation of the objective dunction at each iteration
    """        
    A = np.zeros(size)
    
    # Splitting variable initialisation
    U = M @ A
    V = np.copy(A)
    Z = np.copy(A)

    # Lagrange's multipliers initialisation
    LambdaU = np.zeros(U.shape)
    LambdaV = np.zeros(V.shape)
    LambdaZ = np.zeros(Z.shape)

    # Residuals & objective function initialisation
    norms_primal_U = []  # list of norms of primal residual for U
    objectives = []

    # Pre-computation
    I = np.eye(M.shape[1])
    C = np.linalg.inv(2*I + M.T @ M)  # auxilary variable (pre-computation)

    for _ in range(max_iter):
        A = C @ (M.T @ (U - LambdaU) + (V - LambdaV)) # + (Z - LambdaZ))
        MA = M @ A

        # Splitting variables update
        Nu = MA + LambdaU - 1/rho  # auxilary variable
        U = 0.5 * (Nu + np.sqrt(Nu**2 + 4*Y/rho))
        V = splx.splx_projection(A + LambdaV, r=1)
        # V = np.maximum(A - LambdaV, 0)
        # Z, _ = fbpd.primal_dual_TV(A + LambdaZ, alpha / rho, 0.001)

        # Lagrange's multipliers update
        LambdaU = LambdaU + MA - U
        LambdaV = LambdaV + A - V
        # LambdaZ = LambdaZ + A - Z
        
        # Residuals & objective update
        norms_primal_U.append(np.linalg.norm(M @ A - U, 2))  # residual computation
        objectives.append(objective(M@A, Y))

        clear_output(wait = True)
        print(f"{_+1}    {100*(_+1)/max_iter:.2f} %")

    return A, norms_primal_U, objectives

# %%
