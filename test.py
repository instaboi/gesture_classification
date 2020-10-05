'''
Date: 2020-10-05 15:44:20
LastEditors: Tianling Lyu
LastEditTime: 2020-10-05 16:01:22
FilePath: \gesture_classification\test.py
'''
from argparse import ArgumentParser
import pickle
import time

import mxnet as mx
from mxnet import gluon
from mxnet.ndarray import cast

from network import plain_network
from model import Model
from dataset import DVPickleDataset

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (cast(output.argmax(axis=1), dtype="int64") ==
            label).mean().asscalar()

def test_model(test_file_path, model_path, n_frame):
    # load dataset
    test_dataset = DVPickleDataset(test_file_path, n_frame, 11, False)
    test_data = gluon.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, last_batch="keep")

    # model
    print("Loading network...")
    net = plain_network()
    model = Model(net, load_path=model_path, save_path=".", ctx=mx.gpu())

    # testing
    print("Start testing...")
    test_acc = 0.
    test_time = 0.
    for data, label in test_data:
        begin = time.time()
        out = model.forward(data)
        duration = time.time() - begin
        accu = acc(out.copyto(mx.cpu()), label)
        test_acc += accu
        test_time += duration
    print("Testing finished, acc=%.5f, cost %.5f seconds in average." % (test_acc / len(test_data), test_time / len(test_data)))
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, dest="data_path", default="test.pickle")
    parser.add_argument("--nframe", type=int, dest="n_frame", default=30)
    parser.add_argument("--model-path", type=str, dest="model_path", default="output/augment/RandomResizeCrop/best.model")
    args = parser.parse_args()
    test_model(args.data_path, args.model_path, args.n_frame)
