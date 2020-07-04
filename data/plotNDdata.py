import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from scipy.stats import norm

# data = np.load("n_d_records_test.npy")
# data = np.load("n_d_records.npy")
data = np.load("n_d_records_30iter.npy")
# data = np.load("n_d_records_30iter_small.npy")

print data.shape
print data

# exit()
# ns = range(100, 300, 100)
# ds = range(10, 30, 10)
# ns = range(100, 1000, 100)
# ds = range(10, 100, 10)

#'''
ns = []
ds = []

nn = 100
dd = 5

n_count = 12 #6#10 #15
d_count = 11 #5#12 #17
for i in range(n_count):
    ns.append(nn)
    nn = nn*2

for i in range(d_count):
    ds.append(dd)
    dd = dd*2
print ns, ds
#'''

# data = np.random.random((len(ns), len(ds)))
mat = np.zeros((len(ns), len(ds)))
mat_p = np.zeros((len(ns), len(ds)))
# data = np.loadtxt("d_n_log_data.txt", delimiter=' ')
# data = np.loadtxt("d_n_r10_data.txt", delimiter=' ')

for item in data:
    mat[ns.index(item[0]), ds.index(item[1])] = np.mean(item[2])

    # print(item[3:])
    mu, std = norm.fit(item[3:])
    print(mu, std)
    p = 1.0-norm.cdf(0.5, mu, std)
    mat_p[ns.index(item[0]), ds.index(item[1])] = p

fig, ax = plt.subplots(1,1)
# cax = ax.matshow(mat)
cax = ax.matshow(mat_p, cmap="Reds", vmin=0, vmax=0.05)
fig.colorbar(cax)
ax.set_xticks(np.arange(len(ds)))
ax.set_yticks(np.arange(len(ns)))
ax.set_xticklabels(ds, rotation=45)
ax.set_yticklabels(ns)
plt.xlabel("Dimension")
plt.ylabel("Sample Size")
# plt.show()
# plt.savefig('evaluate_R2.png', bbox_inches='tight')
plt.savefig('evaluate_P.png', bbox_inches='tight')
