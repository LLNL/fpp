#!/usr/bin/env python
#for python2 compactability
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy.stats import norm
import numpy as np
import os, sys
import shutil

class fpp:
    def __init__(self, sparse=False, randomSeed = 1234):
        #### optimize for sparse projection matrix
        self.sparseProj = sparse
        self.classification = False
        self.printOutput = False
        self.randomSeed = randomSeed

    def regress_model(self, x_hat, degree, scope):
        with tf.variable_scope("poly_fn_"+scope):
            wts = []
            if degree==1:
                m = tf.Variable(tf.random_normal([1]), name='slope')
                b = tf.Variable(tf.random_normal([1]), name='bias')
                for j in range(2):
                    Y_pred = tf.add(tf.multiply(x_hat[:,j], m), b)
            else:

                Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
                wts.append(Y_pred)
                for pow_i in range(0, degree):
                    for j in range(2):
                        W = tf.Variable(tf.random_normal([1]), name='weight_{:d}_{:d}'.format(pow_i,j))
                        wts.append(W)
                        Y_pred = tf.add(tf.multiply(tf.pow(x_hat[:,j], pow_i), W), Y_pred)
            return Y_pred,wts

    def weight_variable(self, shape, stddev=0.02, name=None):
        initial = tf.truncated_normal(shape, stddev=stddev)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial,regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))


    def class_model(self, x_hat, scope):
        with tf.variable_scope("logistic_cls_"+scope):
            W = tf.Variable(tf.random_normal([2,1]), name='slope')
            b = tf.Variable(tf.random_normal([1]), name='bias')
            # W = tf.get_variable("W", [2, 1], initializer = tf.contrib.layers.xavier_initializer())
            # b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())
            # Construct model
            # print("x_hat shape:", x_hat.get_shape())
            Z = tf.matmul(x_hat, W)
            # print("Z shape:", Z.get_shape())
            # print("b shape:", b.get_shape())
            # pred = tf.nn.sigmoid(tf.add(Z, b)) # Softmax
            pred = tf.add(Z, b) # Softmax
            # pred = tf.nn.softmax(tf.add(tf.matmul(x_hat, W), b)) # Softmax
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

        tf.reset_default_graph()
        tf.set_random_seed(self.randomSeed)
        np.random.seed(self.randomSeed)


        x_dim = sample.shape[1]
        self.featureSize = x_dim
        self.x = tf.placeholder(tf.float32, shape=[None, x_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_class])
        self._R2 = tf.zeros([self.n_class])

        self.W = self.weight_variable([x_dim,2], stddev=0.02, name="proj")
        _,self.U,_ = tf.linalg.svd(self.W)
        self.x_hat = tf.matmul(self.x,self.U)

        #### add nonlinearility #####
        if self.nonlinear:
            middleLayerSize = self.n_class #if n_class == 2, then use 2 as the middle layer size
            if self.n_class>2:
                middleLayerSize = self.n_class//2
            nl_W1 = tf.Variable(tf.truncated_normal([2,middleLayerSize]), name='NL_slope_1')
            nl_b1 = tf.Variable(tf.zeros([middleLayerSize]), name='NL_bias_1')
            self.nl_x_hat = tf.nn.sigmoid(tf.matmul(self.x_hat, nl_W1) + nl_b1)
            # self.nl_x_hat = tf.nn.relu(tf.matmul(self.x_hat, nl_W1) + nl_b1)

            # nl_W2 = tf.Variable(tf.truncated_normal([self.n_class/3,self.n_class/2]), name='NL_slope_1')
            # nl_b2 = tf.Variable(tf.zeros([self.n_class/2]), name='NL_bias_2')
            # self.nl_x_hat = tf.nn.sigmoid(tf.matmul(self.nl1_x_hat, nl_W2) + nl_b2)

            cW = tf.Variable(tf.truncated_normal([middleLayerSize,self.n_class]), name='slope')
            cb = tf.Variable(tf.zeros([self.n_class]), name='bias')
            self.Y_pred = tf.matmul(self.nl_x_hat, cW) + cb
        else:
            cW = tf.Variable(tf.truncated_normal([2,self.n_class]), name='slope')
            cb = tf.Variable(tf.zeros([self.n_class]), name='bias')
            self.Y_pred = tf.matmul(self.x_hat, cW) + cb


        self.reconstruction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_pred, labels=self.y))


        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.reconstruction_loss)

        # self.saver = tf.train.Saver()
        # self.ckpt = tf.train.get_checkpoint_state('./func_project')
        self.reset()

    ######### the regression setup #######
    def setup(self, sample, f, regressorType="polynominal", degree=4, lr=1e-3, reg_weight=1e-4):
        self.sample = sample
        self.degree = degree
        self.regressorType = regressorType

        if len(f.shape)==1:
            self.f=np.matrix(f).T
        else:
            self.f = f
        self.n_class = self.f.shape[1]

        tf.reset_default_graph()
        tf.set_random_seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        x_dim = sample.shape[1]
        self.x = tf.placeholder(tf.float32, shape=[None, x_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_class])
        self._R2 = [0]*self.n_class

        self.W = self.weight_variable([x_dim,2],stddev=0.02, name="proj")
        _,self.U,_ = tf.linalg.svd(self.W)
        self.x_hat = tf.matmul(self.x,self.U)

        reg = 0
        loss0 = 0
        for j in range(self.n_class):

            if self.regressorType == "polynominal":
                self.Y_pred, self.wts = self.regress_model(self.x_hat,self.degree,"fn_"+str(j))
                # Y_pred,x_hat = regress_model_fcn(x)
                reg += reg_weight*tf.reduce_sum(tf.abs(self.wts))
                # regularization_param = 0
                # print self.y[:,j].shape, self.Y_pred.shape
                # loss0 += tf.reduce_mean(tf.square(self.Y_pred - self.y[:,j]))
                loss0 += tf.reduce_mean(tf.abs(self.Y_pred - self.y[:,j]))
                if self.sparseProj:
                    loss0 += 1e-2*(tf.reduce_sum(tf.abs(self.U[:,0])) + tf.reduce_sum(tf.abs(self.U[:,1])))
                # self.L1 = tf.reduce_mean(tf.abs(self.Y_pred - self.y))

            ###### compute R2 ########
            total_error = tf.reduce_sum(tf.square(tf.subtract(self.y[:,j], tf.reduce_mean(self.y[:,j]))))
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(self.y[:,j], self.Y_pred)))
            self._R2[j] = tf.subtract(1.0, tf.divide(unexplained_error, total_error))

        self.reconstruction_loss = (loss0+reg)/self.n_class

        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.reconstruction_loss)

        self.reset()

    def reset(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



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

                _, self.loss, self.R2, self.projMat = self.sess.run([self.optimizer,self.reconstruction_loss, self._R2, self.U],feed_dict={self.x:batch,self.y:y_batch})
                if self.printOutput:
                    if i%500==0:
                        print('step {:d} reconstruction error: {:.2f}'.format(i, self.loss))
                    if i==step-1:
                        print('step {:d} reconstruction error: {:.2f}'.format(i, self.loss))

    def eval_fit(self):
        y_regress = self.sess.run(self.Y_pred,feed_dict={self.x:self.sample})
        return y_regress

    def predict(self, samples):
        prediction = self.sess.run(self.Y_pred,feed_dict={self.x:samples})
        return prediction


    ####### evaluate global behavior ######
    def eval(self):
        # print('Loss is :',loss)
        x_proj,proj_mat,y_regress, self.loss, self.R2 = self.sess.run([self.x_hat,self.U,self.Y_pred,self.reconstruction_loss, self._R2],feed_dict={self.x:self.sample,self.y:self.f})
        # x_proj,proj_mat,y_regress,reg = self.sess.run([self.x_hat,self.U,self.Y_pred,self.wts],feed_dict={self.x:self.sample})
        # R = [r[0] for r in reg]
        # plt.plot(R,'x-')
        # plt.grid()
        # plt.show()
        return proj_mat, x_proj, self.loss, self.R2
        # x_proj,y_regress = sess.run([x_hat,Y_pred],feed_dict={x:sample})


    ####### estimate p-value ##########
    def p_value(self, iteration=30):
        originalLoss, originalR2 = self.loss = self.sess.run([self.reconstruction_loss, self._R2],feed_dict={self.x:self.sample,self.y:self.f})
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
            randomLoss = self.sess.run([self.reconstruction_loss],feed_dict={self.x:self.sample,self.y:self.f})
            # randomLoss, R2 = self.sess.run([self.reconstruction_loss, self._R2],feed_dict={self.x:self.sample,self.y:self.f})

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
