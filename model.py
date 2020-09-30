'''
Date: 2020-09-28 15:33:01
LastEditors: Tianling Lyu
LastEditTime: 2020-09-29 14:44:14
FilePath: \gesture_classification\model.py
'''

import os
import time

import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import Trainer

class Model:
    def __init__(self, net, load_path=None, save_path="", loss=None, trainer=None, ctx=None):
        if ctx is None:
            self.ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
        else:
            self.ctx = ctx
        self.__net = net
        if load_path is not None:
            self.__net.load_parameters(load_path, ctx=self.ctx)
        else:
            self.__net.initialize(init=init.Xavier(), ctx=self.ctx, force_reinit=True)
        self.__save_path = save_path
        if not os.path.exists(self.__save_path):
            os.makedirs(self.__save_path)
        self.__loss = loss
        self.__trainer = trainer
    
    def load_parameters(self, load_path):
        self.__net.load_parameters(load_path, self.ctx)
        return
    
    def set_loss(self, loss=gluon.loss.SoftmaxCrossEntropyLoss()):
        self.__loss = loss
        return
    
    def set_trainer(self, trainer):
        self.__trainer = trainer
        return
    
    def initialize(self, load_path=None):
        if load_path is not None:
            self.__net.load_parameters(load_path, ctx=self.ctx)
        else:
            self.__net.initialize(init=init.Xavier(), ctx=self.ctx, force_reinit=True)
        return
    
    def train(self, train_data, valid_data, batch_size, n_epoch):
        # perform validation before training the network
        best_loss = 0.
        for data, label in valid_data:
            best_loss += self.__loss(self.__net(data.copyto(self.ctx)), label.copyto(self.ctx)).mean().asscalar()
        best_loss /= len(valid_data)
        best_train = best_loss
        print("Before training: test loss %.3f" % (best_loss))
        self.__net.save_parameters(os.path.join(self.__save_path, "best.model"))
        # start training the network
        for epoch in range(n_epoch):
            train_loss, valid_loss = 0., 0.
            tic = time.time()
            for data, label in train_data:
                # forward + backward
                with autograd.record():
                    output = self.__net(data.copyto(self.ctx))
                    loss = self.__loss(output, label.copyto(self.ctx))
                loss.backward()
                # update parameters
                self.__trainer.step(batch_size)
                # calculate training metrics
                train_loss += loss.mean().asscalar()
            # calculate validation accuracy
            for data, label in valid_data:
                valid_loss += self.__loss(self.__net(data.copyto(self.ctx)), label.copyto(self.ctx)).mean().asscalar()
            train_loss /= len(train_data)
            valid_loss /= len(valid_data)
            print("Epoch %d: loss %.3f, test loss %.3f, in %.1f sec" % (
                    epoch, train_loss, valid_loss, time.time()-tic))
            if train_loss < best_train:
                best_train = train_loss
            if valid_loss < best_loss:
                best_loss = valid_loss
                self.__net.save_parameters(os.path.join(self.__save_path, "best.model"))
                print("\tCurrent best epoch!")
        self.__net.save_parameters(os.path.join(self.__save_path, "last.model"))
        return best_train, best_loss
    
    def test(self, test_data, acc):
        test_loss, test_acc = 0., 0.
        tic = time.time()
        for data, label in test_data:
            test_out = self.__net(data)
            test_loss += self.__loss(test_out, label).mean().asscalar()
            test_acc += acc(test_out, label)
        test_loss /= len(test_data)
        test_acc /= len(test_data)
        print("test loss %.3f, test acc %.3f, in %.1f sec" % (
                test_loss, test_acc, time.time()-tic))
        return test_loss, test_acc
    
    def forward(self, data):
        return self.__net(data)