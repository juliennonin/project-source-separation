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
# R, Nx = 5, 50  # nb of endmembers, size of map
# A = data.generate_abundance_map(R, Nx, thresh=-np.inf, alpha=3)
A, (Nx, Ny, R) = data.fetch_abundance_map()
wavelengths, mat_names, M = data.generate_endmembers(R)
L, N = len(wavelengths), Nx * Ny
COEF = 500
M *= COEF
Y_true = M @ A
poisson = np.vectorize(np.random.poisson)
Y = poisson(Y_true)

N_display =  np.random.randint(N, size=(5))

#%% Display data
%matplotlib qt
fig = plt.figure()
axY = plt.subplot(221)
sY = SliderPlot(axY, Y, title=r"$Y_\lambda$", legend=r"$\lambda$", valinit=91)
axA = plt.subplot(222)
sA = SliderPlot(axA, A, title=r"$A_r$", legend=r"r")
axMA = plt.subplot(223)
sMA = SliderPlot(axMA, Y_true, slider=sY.slider, valinit=91)
axM = plt.subplot(224)
spectrum, = axM.plot(wavelengths, M[:,2])
# plt.title(mat_names[3])
sA.slider.on_changed(lambda r : spectrum.set_ydata(M[:, r]))
plt.show()

#%%
%matplotlib inline
for i, n in enumerate(N_display):
    plt.plot(Y_true[:,n], c=f'C{i}')
    plt.plot(Y[:,n], c=f'C{i}', label=str(n))
plt.legend()
plt.show()

#%% Compute ADMM without regularization term
A_hat, r, F = admm(M, Y, rho=0.1, alpha=1e-3, sigma=0.02, size=(R, N))  # rho=0.1

#%% Display result
%matplotlib qt
fig = plt.figure()
axY = plt.subplot(131)
sY = SliderPlot(axY, Y, title="Y")
axYh = plt.subplot(132)
# sA = SliderPlot(axA, M@A_hat, title="Y hat")
sYh = SliderPlot(axYh, M@A_hat, slider=sY.slider, title="Y hat")
axYt = plt.subplot(133)
sYt = SliderPlot(axYt, Y_true, slider=sY.slider, title="Y true")
plt.show()

#%%
# N_display = np.random.randint(N, size=(5))
for i, n in enumerate(N_display):
    plt.plot((M@A_hat)[:,n], c=f'C{i}', label=str(n))
    plt.plot(Y_true[:,n], '--', c=f'C{i}', alpha=0.7)
plt.margins(x=0)
plt.legend()
plt.show()
#%% Display residuals and objectives
plt.figure()
plt.plot(r)

# plt.plot(F)
plt.show()


# %%
