'''
Date: 2020-09-28 16:33:10
LastEditors: Tianling Lyu
LastEditTime: 2020-09-29 18:26:37
FilePath: \gesture_classification\main.py
'''

from argparse import ArgumentParser

import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon.data.vision import transforms

from network import plain_network
from model import Model
from dataset import DVPickleDataset

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()

def train_and_test_model(train_path, valid_path, test_path, n_frame, batch_size, lr, save_path, n_epoch):
    # dataset
    train_dataset = DVPickleDataset(train_path, n_frame, 11)
    valid_dataset = DVPickleDataset(valid_path, n_frame, 11, False)
    test_dataset = DVPickleDataset(test_path, n_frame, False)
    train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, last_batch="keep")
    valid_data = gluon.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, last_batch="keep")
    test_data = gluon.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, last_batch="keep")

    # model
    net = plain_network()
    net.initialize(init=init.Xavier(), ctx=mx.gpu())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "adam", {'learning_rate': lr})
    model = Model(net, load_path=None, save_path=save_path, loss=loss, trainer=trainer, ctx=mx.gpu())

    # training
    print("Start training...")
    best_loss = model.train(train_data, valid_data, batch_size, n_epoch)
    print("Training finished, best loss=%.3f" % (best_loss))

    # testing
    print("Start testing...")
    test_loss, test_acc = model.test(test_data, acc)
    print("Testing finished, loss=%.3f, acc=%.3f" % (test_loss, test_acc))
    return

def tune_parameters(train_path, valid_path, batch_size, save_path, n_epoch):
    n_frames = [10, 30, 50, 70]
    lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    net = plain_network()
    net.initialize(init=init.Xavier(), ctx=mx.gpu())
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    best_train_loss = 10000000
    best_n_frame = 0
    best_lr = 0

    for n_frame in n_frames:
        # dataset
        train_dataset = DVPickleDataset(train_path, n_frame, 11)
        valid_dataset = DVPickleDataset(valid_path, n_frame, 11, False)
        train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, last_batch="keep")
        valid_data = gluon.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, last_batch="keep")

        for lr in lrs:
            # model
            trainer = gluon.Trainer(net.collect_params(), "adam", {'learning_rate': lr})
            model = Model(net, load_path=None, save_path=save_path, loss=loss, trainer=trainer, ctx=mx.gpu())

            # training
            print("Start training...")
            train_loss, _ = model.train(train_data, valid_data, batch_size, n_epoch)
            print("Current model: n_frame=%d, lr=%f, train_loss=%f" % (n_frame, lr, train_loss))
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_n_frame = n_frame
                best_lr = lr
                print("\tCurrent best model!")
            else:
                print("\tBest model: n_frame=%d, lr=%f, train_loss=%f" % (best_n_frame, best_lr, best_train_loss))
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train-path", type=str, dest="train_path", default="train_pre.pickle")
    parser.add_argument("--valid-path", type=str, dest="valid_path", default="valid_pre.pickle")
    parser.add_argument("--test-path", type=str, dest="test_path", default="test.pickle")
    parser.add_argument("--nframe", type=int, dest="n_frame", default=30)
    parser.add_argument("--batchsize", type=int, dest="batch_size", default=128)
    parser.add_argument("--lr", type=float, dest="lr", default=0.001)
    parser.add_argument("--save-path", type=str, dest="save_path", default="output")
    parser.add_argument("--epoch", type=int, dest="n_epoch", default=100)
    parser.add_argument("--tune", type=bool, dest="tune", default=True)
    args = parser.parse_args()
    if args.tune:
        tune_parameters(args.train_path, args.valid_path, args.batch_size, args.save_path, args.n_epoch)
    else:
        train_and_test_model(args.train_path, args.valid_path, args.test_path, args.n_frame, args.batch_size, args.lr, args.save_path, args.n_epoch)