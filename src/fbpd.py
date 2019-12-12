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

def objective(X, A, Y, lam):
    return 0.5 * np.linalg.norm(A(X) - Y, ord='fro')**2 + lam * TV(X)

#%%
def primal_dual_TV(Y, lam, eps, A=lambda X: X, A_t=lambda X: X):
    print(f'\tdual primal TV with lam={lam}, eps={eps}')
    X = np.copy(Y)
    stopping_crit = eps + 1 
    sigma = lam
    tau = 0.99 / (0.5 + 8*sigma)
    n, m = np.shape(Y)
    Uh, Uv = np.zeros((n,m)), np.zeros((n,m))
    crit = []
    count = 0
    while stopping_crit > eps and count < 300:
        Xold = np.copy(X)  # (Iaux → uk, Iout → uk+1)
        
        # Primal update
        X = X - tau * A_t((A(X) - Y)) - tau * discrete_gradient_adjoint((Uh, Uv))
        np.clip(X, 0, 255, out=X)  # projection
        
        # Dual update
        Vh, Vv = discrete_gradient(2*X - Xold)
        Vh, Vv = Uh + sigma * Vh, Uv + sigma * Vv
        Xaux = np.maximum(np.sqrt(Vh**2 + Vv**2), 1)  # just auxiliare for computation
        Uh = Uh / Xaux
        Uv = Uv / Xaux

        # Criterion
        crit.append(objective(X, A, Y, lam))
        # stopping_crit = np.max(np.abs(X - Xold) / np.abs(Xold))
        if count > 1:
            stopping_crit = np.abs(crit[-1] - crit[-2]) / np.abs(crit[-2])
            # print(count, stopping_crit)
        if count%10==0:
            print('\t', count, stopping_crit)
        count += 1
    print('\t', count, stopping_crit)
    return X, count


# %%