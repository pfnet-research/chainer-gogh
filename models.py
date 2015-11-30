
import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe
from chainer import Variable, optimizers


class NIN:
    def __init__(self, fn="nin_imagenet.caffemodel", alpha=[0,0,1,1], beta=[1,1,1,1]):
        print "load model... %s"%fn
        self.model = caffe.CaffeFunction(fn)
        self.alpha = alpha
        self.beta = beta
    def forward(self, x):
        y0 = F.relu(self.model.conv1(x))
        y1 = self.model.cccp2(F.relu(self.model.cccp1(y0)))
        x1 = F.relu(self.model.conv2(F.average_pooling_2d(F.relu(y1), 3, stride=2)))
        y2 = self.model.cccp4(F.relu(self.model.cccp3(x1)))
        x2 = F.relu(self.model.conv3(F.average_pooling_2d(F.relu(y2), 3, stride=2)))
        y3 = self.model.cccp6(F.relu(self.model.cccp5(x2)))
        x3 = F.relu(getattr(self.model,"conv4-1024")(F.dropout(F.average_pooling_2d(F.relu(y3), 3, stride=2), train=False)))
        return [y0,x1,x2,x3]

class VGG:
    def __init__(self, fn="VGG_ILSVRC_16_layers.caffemodel", alpha=[0,0,1,1], beta=[1,1,1,1]):
        print "load model... %s"%fn
        self.model = caffe.CaffeFunction(fn)
        self.alpha = alpha
        self.beta = beta
    def forward(self, x):
        y1 = self.model.conv1_2(F.relu(self.model.conv1_1(x)))
        x1 = F.average_pooling_2d(F.relu(y1), 2, stride=2)
        y2 = self.model.conv2_2(F.relu(self.model.conv2_1(x1)))
        x2 = F.average_pooling_2d(F.relu(y2), 2, stride=2)
        y3 = self.model.conv3_3(F.relu(self.model.conv3_2(F.relu(self.model.conv3_1(x2)))))
        x3 = F.average_pooling_2d(F.relu(y3), 2, stride=2)
        y4 = self.model.conv4_3(F.relu(self.model.conv4_2(F.relu(self.model.conv4_1(x3)))))
    #    x4 = F.average_pooling_2d(F.relu(y4), 2, stride=2)
    #    y5 = model.conv5_3(F.relu(model.conv5_2(F.relu(model.conv5_1(x4)))))
        return [y1,y2,y3,y4]

class I2V:
    def __init__(self, fn="illust2vec_tag_ver200.caffemodel", alpha=[0,0,0,1,10,100], beta=[0.1,1,1,10,100,1000]):
        print "load model... %s"%fn
        self.model = caffe.CaffeFunction(fn)
        self.alpha = alpha
        self.beta = beta
#        self.pool_func = F.max_pooling_2d
        self.pool_func = F.average_pooling_2d

    def forward(self, x):
        y1 = self.model.conv1_1(x)
        x1 = self.pool_func(F.relu(y1), 2, stride=2)
        y2 = self.model.conv2_1(x1)
        x2 = self.pool_func(F.relu(y2), 2, stride=2)
        y3 = self.model.conv3_2(F.relu(self.model.conv3_1(x2)))
        x3 = self.pool_func(F.relu(y3), 2, stride=2)
        y4 = self.model.conv4_2(F.relu(self.model.conv4_1(x3)))
        x4 = self.pool_func(F.relu(y4), 2, stride=2)
        y5 = self.model.conv5_2(F.relu(self.model.conv5_1(x4)))
        x5 = self.pool_func(F.relu(y5), 2, stride=2)
        y6 = self.model.conv6_4(F.relu(F.dropout(self.model.conv6_3(F.relu(self.model.conv6_2(F.relu(self.model.conv6_1(x5))))),train=False)))
        #x6 = F.average_pooling_2d((y6), y6.data.shape[2], stride=1)
        return [y1,y2,y3,y4,y5,y6]

class GoogLeNet:
    def __init__(self, fn="bvlc_googlenet.caffemodel", alpha=[0,0,0,0,1,10], beta=[0.00005, 5, 50, 50, 5000, 500000]):
        print "load model... %s"%fn
        self.model = caffe.CaffeFunction(fn)
        self.alpha = alpha
        self.beta = beta
#        self.pool_func = F.max_pooling_2d
        self.pool_func = F.average_pooling_2d

    def forward(self, x):
        y1 = self.model['conv1/7x7_s2'](x)
        h = F.relu(y1)
        h = F.local_response_normalization(self.pool_func(h, 3, stride=2), n=5)
        h = F.relu(self.model['conv2/3x3_reduce'](h))
        y2 = self.model['conv2/3x3'](h)
        h = F.relu(y2)
        h = self.pool_func(F.local_response_normalization(h, n=5), 3, stride=2)
        out1 = self.model['inception_3a/1x1'](h)
        out3 = self.model['inception_3a/3x3'](F.relu(self.model['inception_3a/3x3_reduce'](h)))
        out5 = self.model['inception_3a/5x5'](F.relu(self.model['inception_3a/5x5_reduce'](h)))
        pool = self.model['inception_3a/pool_proj'](self.pool_func(h, 3, stride=1, pad=1))
        y3 = F.concat((out1, out3, out5, pool), axis=1)
        h = F.relu(y3)

        out1 = self.model['inception_3b/1x1'](h)
        out3 = self.model['inception_3b/3x3'](F.relu(self.model['inception_3b/3x3_reduce'](h)))
        out5 = self.model['inception_3b/5x5'](F.relu(self.model['inception_3b/5x5_reduce'](h)))
        pool = self.model['inception_3b/pool_proj'](self.pool_func(h, 3, stride=1, pad=1))
        y4 = F.concat((out1, out3, out5, pool), axis=1)
        h = F.relu(y4)

        h = self.pool_func(h, 3, stride=2)

        out1 = self.model['inception_4a/1x1'](h)
        out3 = self.model['inception_4a/3x3'](F.relu(self.model['inception_4a/3x3_reduce'](h)))
        out5 = self.model['inception_4a/5x5'](F.relu(self.model['inception_4a/5x5_reduce'](h)))
        pool = self.model['inception_4a/pool_proj'](self.pool_func(h, 3, stride=1, pad=1))
        y5 = F.concat((out1, out3, out5, pool), axis=1)
        h = F.relu(y5)

        out1 = self.model['inception_4b/1x1'](h)
        out3 = self.model['inception_4b/3x3'](F.relu(self.model['inception_4b/3x3_reduce'](h)))
        out5 = self.model['inception_4b/5x5'](F.relu(self.model['inception_4b/5x5_reduce'](h)))
        pool = self.model['inception_4b/pool_proj'](self.pool_func(h, 3, stride=1, pad=1))
        y6 = F.concat((out1, out3, out5, pool), axis=1)
        h = F.relu(y6)

        return [y1,y2,y3,y4,y5,y6]
