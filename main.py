#%%
%load_ext autoreload
%autoreload 2

import data.generate_data as data
import numpy as np
from utils.slider_plot import SliderPlot
import matplotlib.pyplot as plt
from src.admm import admm, objective
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
fig = plt.figure(figsize=(6, 8))
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

#%% Compute ADMM
A_hat, r, F, maes, cc = admm(M, Y, A, rho=0.1, alpha=1e-2, sigma=0.01, size=(R, N))  # rho=0.1

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
%matplotlib qt
fig = plt.figure()
axY = plt.subplot(121)
sA = SliderPlot(axY, A, title="A true")
axYh = plt.subplot(122)
sYh = SliderPlot(axYh, A_hat, slider=sA.slider, title="A hat")
plt.show()

#%%
%matplotlib inline
# N_display = np.random.randint(N, size=(5))
for i, n in enumerate(N_display):
    plt.plot((M@A_hat)[:,n], c=f'C{i}', label=str(n))
    plt.plot(Y_true[:,n], '--', c=f'C{i}', alpha=0.7)
plt.margins(x=0)
plt.legend()
plt.show()
#%% Display residuals and objectives
%matplotlib inline
plt.figure(figsize=(10,6))

plt.subplot(221)
plt.plot(maes)
plt.ylabel('MAE')

plt.subplot(222)
plt.plot(F)
plt.ylabel(r'objective $\sum(MA - Y\log(MA))$')

plt.subplot(223)
plt.plot(r)
plt.ylabel(r'$\Vert MA-U\Vert_2$')

plt.subplot(224)
plt.plot(cc)
plt.ylabel(r'$\Vert A-V\Vert_2$')

plt.tight_layout()
plt.show()


# %% Test TV update
from src.fbpd import primal_dual_TV_2D
C = np.copy(A_hat)
D = primal_dual_TV_2D(C, 0.01, 0.01, 0.0001)
fig = plt.figure()
axC = plt.subplot(121)
sC = SliderPlot(axC, C, title="before TV")
axD = plt.subplot(122)
sD = SliderPlot(axD, D, slider=sC.slider, title="after TV")

# %%
