#%%
import numpy as np
import utils.splx_projection.splx_projection as splx
import src.fbpd as fbpd

#%%
def real_objective(U, Y):
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

def objective(U, Y):
    return np.sum(-Y * np.log(U) + U)

#%%
def admm(M, Y, rho, alpha, size, eps_abs=1e-2, eps_rel=1e-3, max_iter=25):
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
    V = A
    Z = A

    # Lagrange's multipliers initialisation
    LambdaU = np.zeros(U.shape)
    LambdaV = np.zeros(V.shape)
    LambdaZ = np.zeros(Z.shape)

    # Residuals & objective function initialisation
    norms_primal_U = []  # list of norms of primal residual for U
    objectives = []

    # Pre-computation
    I = np.eye(M.shape[1])
    C = np.linalg.inv(I + M.T @ M)  # auxilary variable (pre-computation)

    iter = 0
    while iter < max_iter:
        # [TODO] : stopping criterion on norm_primal <= eps_primal and norm_dual <= eps_dual
        print(iter)
        A =  C @ (M.T @ (U - LambdaU) + (V - LambdaV) + (Z - LambdaZ))

        # Splitting variables update
        Nu = (M @ A) + LambdaU - 1/rho  # auxilary variable
        U = 0.5 * (Nu + np.sqrt(Nu**2 + (4*Y/rho)))
        V = splx.splx_projection(A + LambdaV, r=1)
        # V = np.maximum(A - LambdaV, 0)
        Z = fbpd.primal_dual_TV_2D(A + LambdaZ, alpha / rho, 0.0001)

        # Lagrange's multipliers update
        LambdaU = LambdaU + M @ A - U
        LambdaV = LambdaV + A - V
        LambdaZ = LambdaZ + A - Z
        
        # Residuals & objective update
        norms_primal_U.append(np.linalg.norm(M @ A - U, 2))  # residual computation
        objectives.append(objective(M@A, Y))

        iter += 1
    return A, norms_primal_U, objectives