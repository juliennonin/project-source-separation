#%%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import src.pidal as pidal

from scipy.ndimage import gaussian_filter
poisson = np.vectorize(np.random.poisson)
# %%
COEFF = 3000
X = plt.imread("data/img/cameraman_crop.png")[:,:,0]
X *= COEFF

# # blurred with Gaussian kernel
Nx, Ny = X.shape
# b = 5
# Nx, Ny = Nx+2*b, Ny+2*b
# Xtilde = np.zeros((Nx, Ny))
# Xtilde[b:-b, b:-b] = X
Xx, Yy = np.meshgrid(np.arange(Nx)-Nx/2, np.arange(Ny)-Ny/2)
h = 1/(2*np.pi) * np.exp(-0.5*(Xx**2 + Yy**2))
D = np.fft.fft2(h)
Y = pidal.K(np.fft.fft2(h), X)
# Y = gaussian_filter(X, 1)

Y = poisson(Y)
# %%
plt.figure(figsize=(6, 8))
plt.subplot(121)
plt.imshow(X, cmap='gray')
plt.subplot(122)
plt.imshow(Y, cmap='gray')

# %%
alpha = 0.008
rho = 60 * alpha / np.max(X)
Z, maes, isnrs = pidal.pidal_tv(Y, D, X, rho, alpha)

# %%
plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(X, cmap="gray")
plt.subplot(132)
plt.imshow(Y, cmap='gray')
plt.subplot(133)
plt.imshow(Z, cmap="gray")


plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(maes)
plt.xlabel('iterations')
plt.ylabel('MAE')

plt.subplot(122)
plt.plot(isnrs)
plt.xlabel('iterations')
plt.ylabel('ISNR (dB)')
plt.tight_layout()
plt.show()

# %%
