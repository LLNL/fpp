import numpy as np
#import matplotlib.pyplot as plt

'''
Generate test data based on given function
'''

class funcPatternGenerator:
    def __init__(self):
        #### fix random seed
        # np.random.seed(1234)
        pass

    def generate2DPattern(self, func, filename, n=3000):
        #### generate 2D pattern #####

        x = np.random.uniform(-1.0,1.0,n)
        y = np.random.uniform(-1.0,1.0,n)
        f = func(x, y)

        ## plot
        self.plot2D(x,y,f)

        paddingDim = self.randDimPadding(n, 3)
        data = np.concatenate((np.matrix(x).T, np.matrix(y).T, paddingDim), axis=1)
        # print data.shape
        rotation = self.randomRotation(5)
        data = data*rotation
        data = np.concatenate( (data, np.matrix(f).T), axis=1 )
        # print data.shape
        np.save(filename, data)
        return data

    def generateRandomPattern(self, n = 1000, dim = 5):
        data = self.randDimPadding(n, dim)
        # print data.shape
        rotation = self.randomRotation(dim)
        data = np.matmul(data,rotation)
        # data = np.concatenate( (data, np.matrix(f).T), axis=1 )
        # print data.shape
        # np.save(filename, data)
        return data

    def generateHDPattern(self, func, filename, n=3000, dim = 30):
        #### generate 2D pattern #####

        x = np.random.uniform(-1.0,1.0,n)
        y = np.random.uniform(-1.0,1.0,n)
        f = func(x, y)

        ## plot
        # self.plot2D(x,y,f)

        paddingDim = self.randDimPadding(n, dim-2)
        data = np.concatenate((np.matrix(x).T, np.matrix(y).T, paddingDim), axis=1)
        # print data.shape
        rotation = self.randomRotation(dim)
        data = data*rotation
        data = np.concatenate( (data, np.matrix(f).T), axis=1 )
        # print data.shape
        np.save(filename, data)
        return data
        # self.plot2D(data[:,0],data[:,1],np.squeeze(f))

    def generate3DPattern(self, func, filename, n=3000, domainRange=1.0):
        #### generate 2D pattern #####

        x = np.random.uniform(-domainRange,domainRange,n)
        y = np.random.uniform(-domainRange,domainRange,n)
        z = np.random.uniform(-domainRange,domainRange,n)
        f = func(x, y, z)

        ## plot
        self.plot2D(x,y,f)
        # self.plot2D(x,z,f)
        # self.plot2D(y,z,f)

        paddingDim = self.randDimPadding(n, 2)
        data = np.concatenate((np.matrix(x).T, np.matrix(y).T, np.matrix(z).T, paddingDim), axis=1)
        # print data.shape
        rotation = self.randomRotation(5)
        data = data*rotation
        data = np.concatenate( (data, np.matrix(f).T), axis=1 )
        # print data.shape
        np.save(filename, data)

    def plot2D(self, x, y, f):
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter(x,y, c=f, s=3, cmap=cm)
        plt.colorbar(sc)
        plt.show()

    def randDimPadding(self, n, d=3):
        #### padding to higher dimension ######
        paddingDim = np.random.uniform(-1.0,1.0,n*d)
        paddingDim = paddingDim.reshape(n, d)
        # print paddingDim.shape
        return paddingDim

    def randomRotation(self, d=5):
        S = np.random.normal(0, 1.0, d*d).reshape( (d,d) )
        Q, R = np.linalg.qr(S)
        T = np.dot(Q, np.diag(np.sign(np.diag(R))))
        return T

    def showHist(self, x):
        plt.hist(x, normed=True, bins=30)
        plt.ylabel('density');

    def addPerturbation(self, domain):

        #### add perturbation ######
        pass


########## 2D f function ##############
def circle(x, y, r=0.6):
    dist = np.square(x) + np.square(y) - r*r
    # print dist.shape
    # return 1.0/np.abs(dist)
    f = np.sqrt(np.sqrt(np.abs(dist)))
    f = np.max(f) - f
    return f

def U(x,y):
    dist = np.abs(y)
    # print dist.shape
    # return 1.0/np.abs(dist)
    f = np.sqrt(np.sqrt(np.abs(dist)))
    f = np.max(f) - f
    return f

def V(x,y):
    dist = np.abs(x)
    # print dist.shape
    # return 1.0/np.abs(dist)
    f = np.sqrt(np.sqrt(np.abs(dist)))
    f = np.max(f) - f
    return f

def cross(x, y):
    dist = np.abs(np.abs(x) - np.abs(y))
    # print dist.shape
    # return 1.0/np.abs(dist)
    f = np.sqrt(np.sqrt(np.abs(dist)))
    f = np.max(f) - f
    return f

def crossV(x, y):
    # print x.shape
    xy = np.concatenate( (np.abs(np.matrix(x)), np.abs(np.matrix(y)) ), axis=0)
    # print xy.shape
    dist = xy.min(axis=0).T
    # print dist.shape
    # return 1.0/np.abs(dist)
    f = np.squeeze(np.array(np.sqrt(np.sqrt(np.abs(dist))) ))
    # print f.shape
    f = np.max(f) - f
    return f

def crossV3D(x, y, z):
    # print x.shape
    xyz = np.concatenate( ( np.abs(np.matrix(x)), np.abs(np.matrix(y)), np.abs(np.matrix(z)) ), axis=0)
    xyz = np.partition(xyz, 2, axis=0)
    dist = np.sqrt(np.power(xyz[0,:],2) + np.power(xyz[1,:],2)).T
    # print dist.shape
    # return 1.0/np.abs(dist)
    f = np.squeeze(np.array(np.sqrt(dist)))
    # print f.shape
    f = np.max(f) - f
    return f

def ackley(x, y, z, d=3):
    domain = np.concatenate( (np.matrix(x), np.matrix(y), np.matrix(z)), axis=0)
    # print "domain", domain.shape
    Sum = np.zeros(x.shape)
    for i in range(d-1):
        theta1 = 6*domain[i, :] - 3
        theta2 = 6*domain[i+1, :] - 3
        # sum -= exp(-0.2) * sqrt(pow(theta1, 2) + pow(theta2, 2)) + 3 * (cos(2 * theta1) + sin(2 * theta2));
        Sum = Sum - np.exp(-0.2) * np.sqrt(np.power(theta1, 2) + np.power(theta2, 2)) + 3 * (np.cos(2 * theta1) + np.sin(2 * theta2));

    Sum = np.squeeze(np.array(Sum.T))
    # print Sum.shape
    return Sum
########### test ############
# gen = funcPatternGenerator()
# gen.generate2DPattern(circle, "data/circle_in_5D_cube.npy")
# gen.generate2DPattern(cross, "data/cross_in_5D_cube.npy")
# gen.generate2DPattern(crossV, "data/crossV_in_5D_cube.npy")
# gen.generate2DPattern(U, "data/U_in_5D_cube.npy")
# gen.generateHDPattern(cross, "data/cross_in_60D.npy",3000,60)
# gen.generateHDPattern(circle, "data/circle_in_300D.npy",3000,300)

# gen.generateHDPattern(circle, "data/circle_in_30D.npy")
# gen.generate2DPattern(V, "data/V_in_5D_cube.npy")
# gen.generate3DPattern(crossV3D, "data/crossV3D_in_5D_cube.npy", n=6000)
# gen.generate3DPattern(ackley, "data/ackley3D_in_5D_cube.npy", n=6000, domainRange=2.0)
