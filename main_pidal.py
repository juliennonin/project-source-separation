#%%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import src.pidal_tv as pidal

# %%
COEFF = 3000
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
