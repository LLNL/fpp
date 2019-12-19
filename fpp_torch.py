#!/usr/bin/env python
#for python2 compactability
from __future__ import division
from __future__ import print_function

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable
from scipy.stats import norm
import numpy as np
import os, sys
import shutil
from torch.autograd import Variable

class fpp:
    def __init__(self, sparse=False, randomSeed = 1234, printOutput = False):
        #### optimize for sparse projection matrix
        self.sparseProj = sparse
        self.printOutput = printOutput
        self.randomSeed = randomSeed
        self.classification = False

    def regress_model(self, x_hat, degree):
        #self.wts = 
        Y_pred = self.wts[-1]
        for pow_i in range(0, degree):
            for j in range(2):
                Y_pred = torch.mul( self.wts[pow_i], torch.pow(x_hat[:,j], pow_i)) + Y_pred
        return Y_pred

    ######### the regression setup #######
    def setup(self, sample, f, regressorType="polynominal", degree=4, lr=1e-3, reg_weight=1e-4):
        self.sample = sample
        self.degree = degree
        self.regressorType = regressorType
        self.reg_weight = reg_weight

        if len(f.shape)==1:
            self.f=np.matrix(f).T
        else:
            self.f = f
        self.n_class = self.f.shape[1]

        # torch.reset_default_graph()
        torch.manual_seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        x_dim = sample.shape[1]
        self.W = Variable(torch.zeros( [x_dim,2] ), requires_grad=True)
        self.wts = Variable(torch.zeros( [degree+1,1] ), requires_grad=True)

        torch.nn.init.normal_(self.W, mean=0, std=0.02)
        torch.nn.init.normal_(self.wts, mean=0.5, std=0.2)

        self.optimizer = torch.optim.Adam([self.W, self.wts], lr=lr)

    def train_regressor(self, x, y):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        self._R2 = [0]*self.n_class
        self.U,_,_ = torch.svd(self.W)
        self.x_hat = torch.matmul(x,self.U)

        
        loss = Variable( torch.FloatTensor([0]))
        reg = Variable( torch.FloatTensor([0]))
        for j in range(self.n_class):
            if self.regressorType == "polynominal":
                self.Y_pred = self.regress_model(self.x_hat,self.degree)
                reg = self.reg_weight*torch.abs(self.wts).sum() + reg
                # regularization_param = 0
                # print y[:,j].shape, self.Y_pred.shape
                # loss0 += torch.mean(torch.sqrt(self.Y_pred - y[:,j]))
                
                loss = torch.mean(torch.abs(self.Y_pred - y[:,j]))

                if self.sparseProj:
                    loss = 1e-2*(torch.abs(self.U[:,0]).sum() + torch.abs(self.U[:,1]).sum() ) + loss
                # self.L1 = torch.mean(torch.abs(self.Y_pred - y))

            ###### compute R2 ########
            total_error = torch.sum(torch.sqrt(torch.sub(y[:,j], torch.mean(y[:,j]))))
            unexplained_error = torch.sqrt(torch.sub(y[:,j], self.Y_pred)).sum()
            self._R2[j] = torch.sub(torch.FloatTensor([1.0]), torch.div(unexplained_error, total_error))

        self.loss = (loss+reg)/self.n_class
        print("loss:", self.loss)
        
        self.optimizer.zero_grad()
        self.loss.backward()

        self.optimizer.step()

    ######### the classification setup #######
    #'''
    def setupMultiClass(self, sample, f, lr=1e-3, reg_weight=1e-5, nonlinear = False):
        self.classification = True
        self.nonlinear = nonlinear
        self.sample = sample
        if len(f.shape)==1:
            self.f=np.matrix(f).T
        else:
            self.f = f
        self.n_class = np.max(self.f)+1
        # self.n_class = self.f.shape[1]
        print("n_class:", self.n_class)

        # torch.reset_default_graph()
        torch.manual_seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        x_dim = sample.shape[1]
        self.featureSize = x_dim
        self._R2 = torch.zeros([self.n_class])

        self.W = Variable(torch.zeros( [x_dim,2] ), requires_grad=True)
        torch.nn.init.normal_(self.W, mean=0, std=0.02)

        if self.nonlinear:
            self.middleLayerSize = self.n_class #if n_class == 2, then use 2 as the middle layer size
            if self.n_class>2:
                self.middleLayerSize = self.n_class//2
            # print("middleLayerSize:", self.middleLayerSize)
            
            self.nl_W1 = Variable(torch.zeros( [2, self.middleLayerSize] ), requires_grad=True)
            self.nl_b1 = Variable(torch.zeros( [self.middleLayerSize] ), requires_grad=True)

            # print("nl_W1:", self.nl_W1.size())

            self.cW = Variable(torch.zeros( [self.middleLayerSize, self.n_class] ), requires_grad=True) #slop
            self.cb = Variable(torch.zeros([self.n_class]), requires_grad=True) #bias

            torch.nn.init.normal_(self.nl_W1, mean=0.0, std=0.1)
            torch.nn.init.normal_(self.cW, mean=0.0, std=0.1)

            # torch.nn.init.normal_(self.nl_b1, mean=0.0, std=0.02)
            # torch.nn.init.normal_(self.cb, mean=0.0, std=0.02)

            self.optimizer = torch.optim.Adam([self.W, self.nl_W1, self.nl_b1, self.cW, self.cb], lr=lr)
        else:
            self.cW = Variable(torch.zeros( [self.middleLayerSize,self.n_class] )) #slop
            self.cb = Variable(torch.zeros([self.n_class])) #bias

            torch.nn.init.normal_(self.cW, mean=0.0, std=0.02)
            # torch.nn.init.normal_(self.cb, mean=0.0, std=0.02)

            self.optimizer = torch.optim.Adam([self.W, self.cW, self.cb], lr=lr)

        self.criteria = torch.nn.CrossEntropyLoss()
        # self.criteria = torch.nn.CrossEntropyLoss(reduction='mean')



    def train_classifier(self, x, y):
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y).squeeze()

        self._R2 = [0]*self.n_class
        self.U,_,_ = torch.svd(self.W)
        self.x_hat = torch.matmul(x,self.U)
        
        #### add nonlinearility #####
        if self.nonlinear:
            # print(self.x_hat.size(), self.nl_W1.size())
            self.nl_x_hat = torch.sigmoid(torch.matmul(self.x_hat, self.nl_W1) + self.nl_b1)
            # self.nl_x_hat = torch.nn.relu(torch.matmul(self.x_hat, self.nl_W1) + self.nl_b1)

            # print(self.nl_x_hat.size(), self.cW.size())
            self.Y_pred = torch.matmul(self.nl_x_hat, self.cW) + self.cb
        else:
            self.Y_pred = torch.matmul(self.x_hat, self.cW) + self.cb

        # print("pred:",self.Y_pred.size(), "label:",y.size())
        self.loss = self.criteria(self.Y_pred, y)

        self.optimizer.zero_grad()
        self.loss.backward()
        # print(self.loss)

        self.optimizer.step()

    def reset(self):
        # self.sess.run(torch.global_variables_initializer())
        pass


    def train(self, epoches=1.0, batchSize = 200, minimalStep = 100):
        self.sampleBuffer = self.sample.copy()
        self.fBuffer = self.f.copy()

        step = int( float(len(self.sampleBuffer)) / float(batchSize) * epoches)
        self.epoches = epoches
        self.batchSize = batchSize

        if step < minimalStep:
            step = minimalStep

        print("total step size:", step)

        #### make sure we have valid batch size ####
        if self.sample.shape[0]<batchSize:
            batchSize = self.sample.shape[0]//2

        for i in range(step):
                # idx = np.random.choice(range(self.sampleBuffer.shape[0]),batchSize,replace=False)
                # batch = self.sampleBuffer[idx,:]
                # # batch = self.sample[idx,:]
                # y_batch = self.f[idx,:]

                start_idx = (i * batchSize) % len(self.sampleBuffer)
                if start_idx + batchSize >= len(self.sampleBuffer):
                    shuffleIndex = np.array(range(len(self.sampleBuffer)))
                    np.random.shuffle(shuffleIndex)
                    self.sampleBuffer = self.sampleBuffer[shuffleIndex]
                    self.fBuffer = self.fBuffer[shuffleIndex]
                    start_idx = 0

                    # print("restart")
                batch = self.sampleBuffer[ start_idx:start_idx+batchSize, :]
                y_batch = self.fBuffer[start_idx:start_idx+batchSize, :]

                if self.classification == False:
                    self.train_regressor(batch, y_batch)
                else:
                    self.train_classifier(batch, y_batch)

                if self.printOutput:
                    if i%500==0:
                        print('step {:d} reconstruction error: {:.2f}'.format(i, float(self.loss) ))
                    if i==step-1:
                        print('step {:d} reconstruction error: {:.2f}'.format(i, float(self.loss) ))


    ####### evaluate global behavior ######
    def eval(self):
        # print('Loss is :',loss)
        return self.U.detach().numpy(), self.x_hat.detach().numpy(), self.loss, self._R2

    ####### estimate p-value ##########
    def p_value(self, iteration=30):
        originalLoss, originalR2 = self.loss = self.sess.run([self.loss, self._R2],feed_dict={self.x:self.sample,y:self.f})
        print("originalLoss:", originalLoss)

        randomLossList = []
        R2List = []
        f_copy = self.f.copy()
        for i in range(iteration):
            f_random = f_copy.copy()

            np.random.shuffle(f_random)
            self.f = f_random

            self.reset()
            #### suppress print out ######
            # sys.stdout = open(os.devnull, 'w')
            self.train(self.epoches, self.batchSize)
            randomLoss = self.sess.run([self.reconstruction_loss],feed_dict={self.x:self.sample,y:self.f})
            # randomLoss, R2 = self.sess.run([self.reconstruction_loss, self._R2],feed_dict={self.x:self.sample,y:self.f})

            # sys.stdout = sys.__stdout__
            if np.isnan(randomLoss):
                continue
            print("index:", i, "randomLoss:", randomLoss)
            #print("index:", i, "randomLoss:", randomLoss, "R2:", R2[0])
            randomLossList.append(randomLoss)
            # R2List.append(R2[0])

        mu, std = norm.fit(randomLossList)
        print("Random mean (Loss), std:", mu, std)
        print("lossList:", randomLossList)
        p = norm.cdf(originalLoss, mu, std)
        print("p-value (Loss):", p)
