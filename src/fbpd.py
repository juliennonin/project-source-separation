#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def discrete_gradient(X):
    n,m = X.shape
    Uh = np.append(X[:,1:] - X[:,:-1], np.zeros((n, 1)), axis=1)
    Uv = np.append(X[1:,:] - X[:-1,:], np.zeros((1, m)), axis=0)
    return Uh, Uv

def discrete_gradient_adjoint(U):
    Uh, Uv = U
    return np.append(np.vstack(-Uh[:,0]), Uh[:,:-1] - Uh[:,1:], axis=1) + \
           np.append([-Uv[0]], Uv[:-1,:] - Uv[1:,:], axis=0)

def total_variation(X):
    Uh, Uv = discrete_gradient(X)
    return np.sum(np.sqrt(Uh**2 + Uv**2))

def objective(X, Y, lam):
    return 0.5 * np.linalg.norm(X - Y, ord='fro')**2 + lam * total_variation(X)

#%%
def primal_dual_TV(Y, lam, tau, eps, max_iter=50):
    # print(f'\tdual primal TV with sigma={sigma}, lambda={lam}, eps={eps}')
    rho = 1.99  # relaxation parameter, in [1, 2[
    sigma = 1/tau/8  # proximal parameter
    # tau = 0.99 / (0.5 + 8*sigma)
    
    X = np.copy(Y)  # Intialization of the solution
    stopping_crit = eps + 1 
    
    Uh, Uv = discrete_gradient(X)
    crit = []
    count = 0
    while stopping_crit > eps and count < max_iter:        
        # Primal update
        X_= X - tau * (X - Y) - tau * discrete_gradient_adjoint((Uh, Uv))

        # Dual update
        Vh, Vv = discrete_gradient(2*X_ - X)
        Vh = Uh + sigma * Vh
        Vv = Uv + sigma * Vv
        aux = np.maximum(np.sqrt(Vh**2 + Vv**2) / lam, 1)  # just auxiliare for computation
        Uh_ = Vh / aux
        Uv_ = Vv / aux

        # Relaxation
        X = X + rho * (X_ - X)
        Uh = Uh + rho * (Uh_ - Uh)
        Uv = Uv + rho * (Uv_ - Uv)

        # Criterion
        crit.append(objective(X, Y, lam))
        # stopping_crit = np.max(np.abs(X - Xold) / np.abs(Xold))
        if count > 1:
            stopping_crit = np.abs(crit[-1] - crit[-2]) / np.abs(crit[-2])
            # print(count, stopping_crit)
        # if count%10==0:
            # print('\t', count, stopping_crit)
        count += 1
    # print('\t', count, stopping_crit)
    return X

# %%
def primal_dual_TV_2D(Y, sigma, lam, eps):
    N = np.shape(Y)[1]
    Nx = int(np.sqrt(N))
    Yy = np.copy(Y)
    for i in range(len(Y)):
        X = primal_dual_TV(Y[i].reshape((Nx, -1)).T, sigma, lam, eps)
        Yy[i] = np.hstack(X.T)
    return Yy

# %%
