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

########## MNIST dataset ##########
'''
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape( (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape( (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
sample = X_train
f = (y_train)
print (np.min(f), np.max(f), f.shape, sample.shape)
'''

### setup fpp input ####
model = fpp(printOutput=True)
model.setup(sample, f, degree=3)

### training ####
start = time.time()
model.train(50, 50) ### circle
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
