'''
Date: 2020-09-28 15:33:01
LastEditors: Tianling Lyu
LastEditTime: 2020-10-01 18:22:01
FilePath: \gesture_classification\model.py
'''

import os
import time

import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.ndarray import round, transpose
from mxnet.gluon import Trainer
from mxboard import SummaryWriter

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
    
    def train(self, train_data, valid_data, batch_size, n_epoch, acc):
        # perform validation before training the network
        best_loss = 0.
        global_step = 0
        for data, label in valid_data:
            data = transpose(data, (0, 3, 1, 2))
            best_loss += self.__loss(self.__net(data.copyto(self.ctx)), label.copyto(self.ctx)).mean().asscalar()
        best_loss /= len(valid_data)
        best_train = best_loss
        print("Before training: test loss %.3f" % (best_loss))
        self.__net.save_parameters(os.path.join(self.__save_path, "best.model"))
        sw = SummaryWriter(logdir='./logs')
        # start training the network
        for epoch in range(n_epoch):
            train_loss, valid_loss, valid_acc = 0., 0., 0.
            tic = time.time()
            for data, label in train_data:
                # forward + backward
                data = data.copyto(self.ctx)
                data = transpose(round(data),(0, 3, 1, 2))
                with autograd.record():
                    output = self.__net(data)
                    loss = self.__loss(output, label.copyto(self.ctx))
                loss.backward()
                if global_step % 1000 == 0:
                    sw.add_histogram(tag="loss", values=loss, bins=200, global_step=global_step)
                    sw.add_histogram(tag="output", values=output, bins=200, global_step=global_step)
                # update parameters
                self.__trainer.step(batch_size)
                # calculate training metrics
                train_loss += loss.mean().asscalar()
                global_step += 1
            # calculate validation accuracy
            for data, label in valid_data:
                data = data.copyto(self.ctx)
                data = transpose(data, (0, 3, 1, 2))
                valid_out = self.__net(data)
                label = label.copyto(self.ctx)
                valid_loss += self.__loss(valid_out, label).mean().asscalar()
                valid_acc += acc(valid_out, label)
            train_loss /= len(train_data)
            valid_loss /= len(valid_data)
            valid_acc /= len(valid_data)
            print("Epoch %d: loss %.5f, test loss %.5f, test acc %.5f, in %.1f sec" % (
                    epoch, train_loss, valid_loss, valid_acc, time.time()-tic))
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
            data = data.copyto(self.ctx)
            data = transpose(data, (0, 3, 1, 2))
            label = label.copyto(self.ctx)
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
