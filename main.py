#%%
import data.generate_data as data
import numpy as np

#%%
A = data.generate_img_map()
R, N = A.shape
COEF = 10
M = np.eye(R)



# %%
