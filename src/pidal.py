import numpy as np

from IPython.display import clear_output
from src.fbpd import total_variation, primal_dual_TV

def K(D, X):
    """KX"""
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(X)*D)))
    
def KT(D, X):
    """K^TX"""
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(X)*D)))

def pidal_tv(Y, D, X, rho, alpha=0.008, max_iter=200):
    maes = []
    isnrs = []
    n = Y.shape[0]
    norm2 = np.linalg.norm(Y - X, 2)**2

    # Init
    U1 = np.copy(Y)
    U2 = np.copy(Y)
    U3 = np.copy(Y)

    D1 = np.zeros(Y.shape)
    D2 = np.zeros(Y.shape)
    D3 = np.zeros(Y.shape)

    def C(D, X):
        """(K^* K + 2I)^{-1}X"""
        return np.abs(np.fft.ifft2(np.fft.fft2(X) / (np.abs(D)**2 + 2)))

    for _ in range(max_iter):
        Zeta1 = U1 + D1
        Zeta2 = U2 + D2
        Zeta3 = U3 + D3
        Gamma = KT(D, Zeta1) + Zeta2 + Zeta3
        Z = C(D, Gamma)

        # Splitting variables update
        Nu = K(D, Z) - D1 - 1/rho
        U1 = 0.5 * (Nu + np.sqrt(Nu**2 + 4*Y/rho))

        U2 = primal_dual_TV(Z - D2, 0.01, alpha/rho, 0.001)

        U3 = np.copy(Z - D3)
        U3[U3<0] = 0

        # Lagrange's multipliers update
        D1 = D1 - (K(D, Z) - U1)
        D2 = D2 - (Z - U2)
        D3 = D3 - (Z - U3)

        maes.append(np.linalg.norm(Z - X, 1) / n)
        isnrs.append(10*np.log10(norm2 / np.linalg.norm(Z-X, 2)**2))
        clear_output(wait = True)
        print(f"{_+1}    {100*(_+1)/max_iter:.2f} %")

    return Z, maes, isnrs