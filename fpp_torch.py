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
        wts = []
        if degree==1:
            m = Variable(torch.FloatTensor([1]))
            b = Variable(torch.FloatTensor([1]))
            for j in range(2):
                Y_pred = torch.add(torch.multiply(x_hat[:,j], m), b)
        else:
            Y_pred = Variable(torch.FloatTensor([1]))
            wts.append(Y_pred)
            for pow_i in range(0, degree):
                for j in range(2):
                    W = Variable(torch.FloatTensor([1]))
                    wts.append(W)
                    Y_pred = torch.add(torch.mul(torch.pow(x_hat[:,j], pow_i), W), Y_pred)
        return Y_pred,wts

    def weight_variable(self, shape, stddev=0.02, name=None):
        initial = torch.truncated_normal(shape, stddev=stddev)

        return Variable(initial)

    def class_model(self, x_hat, scope):
        W = Variable(torch.FloatTensor( (2,1) ))
        b = Variable(torch.FloatTensor( (1) ))
        # Construct model
        Z = torch.matmul(x_hat, W)
        pred = torch.add(Z, b) # Softmax
        return pred

    ######### the classification setup #######
    def setupMultiClass(self, sample, f, lr=1e-3, reg_weight=1e-5, nonlinear = False):
        self.classification = True
        self.nonlinear = nonlinear
        self.sample = sample
        if len(f.shape)==1:
            self.f=np.matrix(f).T
        else:
            self.f = f
        self.n_class = self.f.shape[1]

        # torch.reset_default_graph()
        torch.set_random_seed(self.randomSeed)
        np.random.seed(self.randomSeed)


        x_dim = sample.shape[1]
        self.featureSize = x_dim
        self.x = torch.placeholder(torch.float32, shape=[None, x_dim])
        y = torch.placeholder(torch.float32, shape=[None, self.n_class])
        self._R2 = torch.zeros([self.n_class])

        self.W = torch.FloatTensor( (x_dim,2) )
        torch.nn.init.normal_(self.W, mean=0, std=0.02)

        _,self.U,_ = torch.svd(self.W)
        self.x_hat = torch.matmul(self.x,self.U)

        #### add nonlinearility #####
        if self.nonlinear:
            middleLayerSize = self.n_class #if n_class == 2, then use 2 as the middle layer size
            if self.n_class>2:
                middleLayerSize = self.n_class//2
            nl_W1 = Variable(torch.FloatTensor( (2,middleLayerSize) ))
            nl_b1 = Variable(torch.zeros( (middleLayerSize) ))
            self.nl_x_hat = torch.nn.sigmoid(torch.matmul(self.x_hat, nl_W1) + nl_b1)
            # self.nl_x_hat = torch.nn.relu(torch.matmul(self.x_hat, nl_W1) + nl_b1)

            # nl_W2 = Variable(torch.truncated_normal([self.n_class/3,self.n_class/2]), name='NL_slope_1')
            # nl_b2 = Variable(torch.zeros([self.n_class/2]), name='NL_bias_2')
            # self.nl_x_hat = torch.nn.sigmoid(torch.matmul(self.nl1_x_hat, nl_W2) + nl_b2)

            cW = Variable(torch.truncated_normal([middleLayerSize,self.n_class]), name='slope')
            cb = Variable(torch.zeros([self.n_class]), name='bias')
            self.Y_pred = torch.matmul(self.nl_x_hat, cW) + cb
        else:
            cW = Variable(torch.truncated_normal([2,self.n_class]), name='slope')
            cb = Variable(torch.zeros([self.n_class]), name='bias')
            self.Y_pred = torch.matmul(self.x_hat, cW) + cb


        self.reconstruction_loss = torch.reduce_mean(torch.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_pred, labels=y))


        self.optimizer = torch.train.AdamOptimizer(lr).minimize(self.reconstruction_loss)

        # self.saver = torch.train.Saver()
        # self.ckpt = torch.train.get_checkpoint_state('./func_project')


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
        # print("x_dim:", x_dim)
        self.W = Variable(torch.zeros( [x_dim,2] ), requires_grad=True)
        # print(self.W, self.W.size())
        torch.nn.init.normal_(self.W, mean=0, std=0.02)

        self.optimizer = torch.optim.Adam([self.W], lr=lr)

    def train_regressor(self, x, y):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        self._R2 = [0]*self.n_class
        # print(self.W, self.W.size())
        self.U,_,_ = torch.svd(self.W) # U, S, V
        self.x_hat = torch.matmul(x,self.U)

        reg = 0
        loss0 = 0
        for j in range(self.n_class):
            if self.regressorType == "polynominal":
                self.Y_pred, self.wts = self.regress_model(self.x_hat,self.degree)
                # Y_pred,x_hat = regress_model_fcn(x)
                # reg += self.reg_weight*torch.abs(self.wts).sum()
                # regularization_param = 0
                # print y[:,j].shape, self.Y_pred.shape
                # loss0 += torch.reduce_mean(torch.sqrt(self.Y_pred - y[:,j]))
                loss0 += torch.mean(torch.abs(self.Y_pred - y[:,j]))
                if self.sparseProj:
                    loss0 += 1e-2*(torch.abs(self.U[:,0]).sum() + torch.abs(self.U[:,1]).sum() )
                # self.L1 = torch.reduce_mean(torch.abs(self.Y_pred - y))

            ###### compute R2 ########
            total_error = torch.sum(torch.sqrt(torch.sub(y[:,j], torch.mean(y[:,j]))))
            unexplained_error = torch.sqrt(torch.sub(y[:,j], self.Y_pred)).sum()
            self._R2[j] = torch.sub(1.0, torch.div(unexplained_error, total_error))

        self.reconstruction_loss = (loss0+reg)/self.n_class

        self.reconstruction_loss.backward()
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

                self.train_regressor(batch, y_batch)

                if self.printOutput:
                    if i%500==0:
                        print('step {:d} reconstruction error: {:.2f}'.format(i, self.loss))
                    if i==step-1:
                        print('step {:d} reconstruction error: {:.2f}'.format(i, self.loss))


    ####### evaluate global behavior ######
    def eval(self):
        # print('Loss is :',loss)
        return self.U, self.x_hat, self.reconstruction_loss, self._R2

    ####### estimate p-value ##########
    def p_value(self, iteration=30):
        originalLoss, originalR2 = self.loss = self.sess.run([self.reconstruction_loss, self._R2],feed_dict={self.x:self.sample,y:self.f})
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
