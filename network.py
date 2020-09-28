'''
Date: 2020-09-27 16:04:49
LastEditors: Tianling Lyu
LastEditTime: 2020-09-28 16:21:13
FilePath: \gesture_classification\network.py
'''

from mxnet.gluon import nn

# network defined in "A Low Power, Fully Event-Based Gesture Recognition System"
def plain_network():
    net = nn.Sequential()
    # Add a sequence of layers.
    net.add(# Similar to Dense, it is not necessary to specify the input channels
            # by the argument `in_channels`, which will be  automatically inferred
            # in the first forward pass. Also, we apply a relu activation on the
            # output. In addition, we can use a tuple to specify a  non-square
            # kernel size, such as `kernel_size=(2,4)`
            nn.Conv2D(channels=12, kernel_size=3, strides=2, groups=1, activation='relu'), #1

            nn.Conv2D(channels=252, kernel_size=4, strides=2, groups=2, activation='relu'), #2
            nn.Conv2D(channels=256, kernel_size=1, strides=1, groups=2, activation='relu'), #3
            nn.Conv2D(channels=256, kernel_size=2, strides=2, groups=2, activation='relu'), #4

            nn.Conv2D(channels=512, kernel_size=3, strides=1, groups=32, activation='relu'), #5
            nn.Conv2D(channels=512, kernel_size=1, strides=1, groups=4, activation='relu'), #6
            nn.Conv2D(channels=512, kernel_size=1, strides=1, groups=4, activation='relu'), #7
            nn.Conv2D(channels=512, kernel_size=1, strides=1, groups=4, activation='relu'), #8

            nn.Conv2D(channels=512, kernel_size=2, strides=2, groups=16, activation='relu'), #9
            nn.Conv2D(channels=1024, kernel_size=3, strides=1, groups=64, activation='relu'), #10
            nn.Conv2D(channels=1024, kernel_size=1, strides=1, groups=8, activation='relu'), #11
            nn.Conv2D(channels=1024, kernel_size=1, strides=1, groups=8, activation='relu'), #12

            nn.Conv2D(channels=1024, kernel_size=2, strides=2, groups=32, activation='relu'), #13
            nn.Conv2D(channels=1024, kernel_size=1, strides=1, groups=8, activation='relu'), #14
            nn.Conv2D(channels=968, kernel_size=1, strides=1, groups=8, activation='relu'), #15
            nn.Conv2D(channels=2640, kernel_size=1, strides=1, groups=8, activation='relu'), #16
            # max pooling layer into the 2-D shape: (x.shape[0], x.size/x.shape[0])
            nn.Dense(n_gesture)
        )
    return net