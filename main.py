#%%
import data.generate_data as data
import numpy as np
from utils.slider_plot import SliderPlot
import matplotlib.pyplot as plt
import utils.splx_projection.splx_projection as splx
from src.admm import admm
%matplotlib qt

#%%
A = data.generate_img_map()
R, N = A.shape
L = R
COEF = 30
M = np.eye(R)
Y = data.generate_observation(M, A)

#%%
fig = plt.figure()
axY = plt.subplot(121)
sY = SliderPlot(axY, Y, title="Y")
axA = plt.subplot(122)
sA = SliderPlot(axA, A, title="A")
# axZ = plt.subplot(133)
# sZ = SliderPlot(axZ, M)
plt.show()

#%%
r, Ak, F = admm(M, Y, 1e-4, (R, N))

# %%
k = 0
fig = plt.figure(figsize=(10, 6))
plt.subplot('131')
plt.imshow(A[k].reshape((int(np.sqrt(N)), -1)).T, cmap='gray')
ax = plt.subplot('132')

s = SliderPlot(ax, Ak[:,k,:], legend='k', valinit=100)
plt.subplot('133')
plt.imshow(Y[k].reshape((int(np.sqrt(N)), -1)).T, cmap='gray')
plt.show()


# %%
plt.figure()
plt.imshow((Y[k]/COEF - Ak[-1,k]).reshape((int(np.sqrt(N)), -1)).T)
plt.colorbar()
# %%
plt.plot(r)

# %%
