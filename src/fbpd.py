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

def TV(X):
    Uh, Uv = discrete_gradient(X)
    return np.sum(np.sqrt(Uh**2 + Uv**2))

def objective(X, Y, lam):
    return 0.5 * np.linalg.norm(X - Y, ord='fro')**2 + lam * TV(X)

#%%
def primal_dual_TV(Y, lam, tol, max_iter=300):
    print(f'\tdual primal TV with lam={lam}, tol={tol}')
    loss = np.inf
    sigma = lam
    tau = 0.99 / (0.5 + 8*sigma)
    iter = 0
    
    n, m = np.shape(Y)
    X = np.copy(Y)
    # Uh, Uv = np.zeros((n,m)), np.zeros((n,m))
    Uh, Uv = discrete_gradient(X)
    while loss > tol and iter < max_iter:
        Xold = np.copy(X)  # (Xold → uk, X → uk+1)
        
        # Primal update
        X = X - tau * (X - Y) - tau * discrete_gradient_adjoint((Uh, Uv))
        
        # Dual update
        Vh, Vv = discrete_gradient(2*X - Xold)
        Vh, Vv = Uh + sigma * Vh, Uv + sigma * Vv
        aux = np.maximum(np.sqrt(Vh**2 + Vv**2), 1)  # just auxiliare for computation
        Uh = Uh / aux
        Uv = Uv / aux

        # Criterion
        loss = np.linalg.norm(X - Xold, 2) / np.linalg.norm(Xold, 2)
        
        # [Temp] Debugging
        if iter%10==0:
            print('\t', iter, loss)
        
        iter += 1
    print('\t', iter, loss)
    return X, iter

# %%
def primal_dual_TV_2D(Y, *args):
    R, N = np.shape(Y)
    Nx = int(np.sqrt(N))
    for i in range(len(Y)):
        X, _ = primal_dual_TV(Y[i].reshape((Nx, -1)).T, *args)
        Y[i] = np.hstack(X.T)
    return Y

# %%
