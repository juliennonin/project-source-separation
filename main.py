#%%
%load_ext autoreload
%autoreload 2

import data.generate_data as data
import numpy as np
from utils.slider_plot import SliderPlot
import matplotlib.pyplot as plt
from src.admm import admm
# %matplotlib qt

#%%
np.random.seed(179)
R, Nx = 5, 50  # nb of endmembers, size of map
A = data.generate_abundance_map(R, Nx)
wavelengths, mat_names, M = data.generate_endmembers(R)
L, N = len(wavelengths), Nx * Nx
COEF = 500
M *= COEF
Y = data.generate_observation(M, A)

#%% Display data
# %matplotlib qt
# fig = plt.figure()
# axY = plt.subplot(221)
# sY = SliderPlot(axY, Y, title=r"$Y_\lambda$", legend=r"$\lambda$", valinit=91)
# axA = plt.subplot(222)
# sA = SliderPlot(axA, A, title=r"$A_r$", legend=r"r")
# axMA = plt.subplot(223)
# sMA = SliderPlot(axMA, M @ A, slider=sY.slider, valinit=91)
# axM = plt.subplot(224)
# spectrum, = axM.plot(wavelengths, M[:,3])
# # plt.title(mat_names[3])
# sA.slider.on_changed(lambda r : spectrum.set_ydata(M[:, r]))
# plt.show()

#%% Compute ADMM without regularization term
A_hat, r, F = admm(M, Y, 1e-3, 1e-5, (R, N))

#%% Display result
fig = plt.figure()
axY = plt.subplot(121)
sY = SliderPlot(axY, Y, title="Y")
axA = plt.subplot(122)
sA = SliderPlot(axA, A_hat, title="A")
plt.show()

#%% Display residuals and objectives
plt.plot(r)
plt.plot(F)


# %%
