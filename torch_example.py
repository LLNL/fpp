from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from fpp_torch import *


########## circle synthetic data #########

xsample = np.load('data/circle_in_5D_cube.npy')
sample = xsample[:,:-1]
f = xsample[:,-1]
print("sample:", sample.shape)

### setup fpp input ####
model = fpp()
model.setup(sample, f, degree=3)

### training ####
start = time.time()
model.train(100, 50) ### circle
end = time.time()
print("timing:", end-start)

from pylab import rcParams
rcParams['figure.figsize'] = 6, 5
proj_mat, embedding, loss, R2 = model.eval()
proj_mat = proj_mat.detach().numpy()
embedding = np.matmul(sample, proj_mat)
print("embedding:", embedding.shape)
plt.scatter(embedding[:,0], embedding[:,1], c=f, cmap="Spectral", alpha=0.8, s=8)
plt.colorbar()
plt.show()
