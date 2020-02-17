#%%
import numpy as np
import matplotlib.pyplot as plt
from utils.slider_plot import SliderPlot
from utils.gaussian_random_fields.gaussian_random_fields import gaussian_random_field

import scipy.io as sio
import matplotlib.image as mpimg
from os import listdir
#%%
def generate_img_map():
    """[TODO] check size of img"""
    path = "data/img"
    size = 256
    files = listdir(path)
    A = np.zeros((len(files), size * size))
    for i, filename in enumerate(files):
        img = mpimg.imread(path + '/' + filename)
        if img.ndim == 3:  # RGB img
            img = np.mean(img, axis=2)  # convert into greyscale img
        A[i] = img.T.reshape((-1))
    return A

def generate_abundance_map(R, size, thresh=-np.inf, alpha=3):
    N = size * size
    A = np.zeros((R, N))

    # generating R - 1 maps
    for r in range(R - 1):
        abundancy_map = gaussian_random_field(alpha=alpha, size=size)
        A[r] = abundancy_map.T.reshape((-1))
        A[r][A[r] < thresh] = 0
    
    A = A - np.min(A)  # translate to have non-negative coefficients
    
    # generate the last map
    last_map = np.sum(A, axis=0) - A[-1]
    last_map = max(last_map) - last_map
    A[-1] = last_map

    A = A / np.sum(A, axis=0)  # normalize the map

    assert np.all(A >= 0), "All coefficients must be non negative"
    assert np.all(1 - np.sum(A, axis=0) <= 1e-10), "The sum of each column must be equal to 1"
    return A

def fetch_abundance_map(file='data/remote_sensing/madonna_vca.mat'):
    data = sio.loadmat(file)
    A = data['A_VCA']
    Nx, Ny, R = A.shape
    return A.reshape(-1, R).T, (Nx, Ny, R) 

def fetch_endmembers():
    file = 'data/remote_sensing/USGS_1995_Library.mat'
    data = sio.loadmat(file)
    names = data['names']  # material name of size R
    names = ["".join(chr(e) for e in names[i]).strip() for i in range(3, len(names))]
    wavelengths = data["datalib"][:,0]  # size L
    endmembers = data['datalib'][:,3:]  # endmember matrix L×R
    return wavelengths, names, endmembers

def generate_endmembers(R, seed=None):
    """Choose randomly R spectra in USGS collection"""
    wavelengths, names, endmembers = fetch_endmembers()
    _, R_tot = np.shape(endmembers)
    assert R < R_tot, f"R cannot take a larger value than {R_tot}."
    rd = np.random.RandomState(seed)
    r = rd.choice(R_tot, R, replace=False)  # randomly select R indices
    print(r)
    return wavelengths, [names[i] for i in r], endmembers[:, r]

def generate_observation(M, A):
    Lambda = M @ A
    poisson = np.vectorize(np.random.poisson)
    return poisson(Lambda)

# %%
if __name__ == '__main__':
    # %matplotlib qt

    A = generate_img_map()
    B = generate_abundance_map(3, 100)

    fig = plt.figure(figsize=(10, 6))
    axA = plt.subplot('121')
    axB = plt.subplot('122')

    sA = SliderPlot(axA, A)
    sB = SliderPlot(axB, B)
    plt.show()

# %%
