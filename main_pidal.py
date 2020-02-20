#%%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import src.pidal as pidal

# %%
COEFF = 100
X = plt.imread("data/img/cameraman_crop.png")[:,:,0]
X *= COEFF
poisson = np.vectorize(np.random.poisson)
Y = poisson(X)
# %%
plt.figure(figsize=(6, 8))
plt.subplot(121)
plt.imshow(X, cmap='gray')
plt.subplot(122)
plt.imshow(Y, cmap='gray')

# %%
Z, maes, isnrs = pidal.pidal_tv(Y, np.eye(Y.shape[0]), X, 0.1)

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
