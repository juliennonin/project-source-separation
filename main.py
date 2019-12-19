#%%
import data.generate_data as data
import numpy as np
from utils.slider_plot import SliderPlot
import matplotlib.pyplot as plt
from src.admm import admm
# %matplotlib qt

#%%
A = data.generate_img_map()
R, N = A.shape
L = R
COEF = 30
M = COEF * np.eye(R)
# M = 20 * np.random.rand(L, R)
Y = data.generate_observation(M, A)

#%% Display data
# fig = plt.figure()
# axY = plt.subplot(121)
# sY = SliderPlot(axY, Y, title="Y")
# axA = plt.subplot(122)
# sA = SliderPlot(axA, A, title="A")
# # axZ = plt.subplot(133)
# # sZ = SliderPlot(axZ, M)
# plt.show()

#%% Compute ADMM without regularization term
A_hat, r, F = admm(M, Y, 1e-3, 1e-5,(R, N))

#%% Display result
fig = plt.figure()
axY = plt.subplot(121)
sY = SliderPlot(axY, Y, title="Y")
axA = plt.subplot(122)
sA = SliderPlot(axA, A_hat, title="A")

#%% Display residuals and objectives
plt.figure()
plt.plot(r)
plt.plot(F)
plt.show()


# %%
