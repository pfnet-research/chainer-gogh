
import argparse
import os
import sys

import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe
from chainer import Variable, optimizers

import pickle


def subtract_mean(x0):
    x = x0.copy()
    x[0,0,:,:] -= 104
    x[0,1,:,:] -= 117
    x[0,2,:,:] -= 123
    return x
def add_mean(x0):
    x = x0.copy()
    x[0,0,:,:] += 104
    x[0,1,:,:] += 117
    x[0,2,:,:] += 123
    return x


def image_resize(img_file, width):
    gogh = Image.open(img_file)
    orig_w, orig_h = gogh.size[0], gogh.size[1]
    if orig_w>orig_h:
        new_w = width
        new_h = width*orig_h/orig_w
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1,3,new_h,new_w))
        print("image resized to: ", gogh.shape)
        hoge= np.zeros((1,3,width,width), dtype=np.float32)
        hoge[0,:,width-new_h:,:] = gogh[0,:,:,:]
        gogh = subtract_mean(hoge)
    else:
        new_w = width*orig_w/orig_h
        new_h = width
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1,3,new_h,new_w))
        print("image resized to: ", gogh.shape)
        hoge= np.zeros((1,3,width,width), dtype=np.float32)
        hoge[0,:,:,width-new_w:] = gogh[0,:,:,:]
        gogh = subtract_mean(hoge)
    return xp.asarray(gogh), new_w, new_h

def save_image(img, width, new_w, new_h, it):
    def to_img(x):
        im = np.zeros((new_h,new_w,3))
        im[:,:,0] = x[2,:,:]
        im[:,:,1] = x[1,:,:]
        im[:,:,2] = x[0,:,:]
        def clip(a):
            return 0 if a<0 else (255 if a>255 else a)
        im = np.vectorize(clip)(im).astype(np.uint8)
        Image.fromarray(im).save(args.out_dir+"/im_%05d.png"%it)

    if args.gpu>=0:
        img_cpu = add_mean(img.get())
    else:
        img_cpu = add_mean(img)
    if width==new_w:
        to_img(img_cpu[0,:,width-new_h:,:])
    else:
        to_img(img_cpu[0,:,:,width-new_w:])


def nin_forward(x):
    y0 = F.relu(model.conv1(x))
    y1 = model.cccp2(F.relu(model.cccp1(y0)))
    x1 = F.relu(model.conv2(F.average_pooling_2d(F.relu(y1), 3, stride=2)))
    y2 = model.cccp4(F.relu(model.cccp3(x1)))
    x2 = F.relu(model.conv3(F.average_pooling_2d(F.relu(y2), 3, stride=2)))
    y3 = model.cccp6(F.relu(model.cccp5(x2)))
    x3 = F.relu(getattr(model,"conv4-1024")(F.dropout(F.average_pooling_2d(F.relu(y3), 3, stride=2), train=False)))
    return [y0,x1,x2,x3]

def vgg_forward(x):
    y1 = model.conv1_2(F.relu(model.conv1_1(x)))
    x1 = F.average_pooling_2d(F.relu(y1), 2, stride=2)
    y2 = model.conv2_2(F.relu(model.conv2_1(x1)))
    x2 = F.average_pooling_2d(F.relu(y2), 2, stride=2)
    y3 = model.conv3_3(F.relu(model.conv3_2(F.relu(model.conv3_1(x2)))))
    x3 = F.average_pooling_2d(F.relu(y3), 2, stride=2)
    y4 = model.conv4_3(F.relu(model.conv4_2(F.relu(model.conv4_1(x3)))))
#    x4 = F.average_pooling_2d(F.relu(y4), 2, stride=2)
#    y5 = model.conv5_3(F.relu(model.conv5_2(F.relu(model.conv5_1(x4)))))
    return [y1,y2,y3,y4]



def get_matrix(y):
    ch = y.data.shape[1]
    wd = y.data.shape[2]
    gogh_y = F.reshape(y, (ch,wd**2))
    gogh_matrix = F.matmul(gogh_y, gogh_y, transb=True)/np.float32(ch*wd**2)
    return gogh_matrix



class Clip(chainer.Function):
    def forward(self, x):
        x = x[0]
        ret = cuda.elementwise(
            'T x','T ret',
            '''
                ret = x<-100?-100:(x>100?100:x);
            ''','clip')(x)
        return ret

def generate_image(img_orig, img_style, width, nw, nh, max_iter, lr, alpha, beta, img_gen=None):
    mid_orig = nin_forward(Variable(img_orig))
    style_mats = [get_matrix(y) for y in nin_forward(Variable(img_style))]

    if img_gen is None:
        if args.gpu >= 0:
            img_gen = xp.random.uniform(-20,20,(1,3,width,width),dtype=np.float32)
        else:
            img_gen = np.random.uniform(-20,20,(1,3,width,width)).astype(np.float32)
    x = Variable(img_gen)
    xg = xp.zeros_like(x.data)
    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup((img_gen,xg))
    for i in range(max_iter):

        x = Variable(img_gen)
        y = nin_forward(x)

        optimizer.zero_grads()
        L = Variable(xp.zeros((), dtype=np.float32))
        for l in range(4):
            ch = y[l].data.shape[1]
            wd = y[l].data.shape[2]
            gogh_y = F.reshape(y[l], (ch,wd**2))
            gogh_matrix = F.matmul(gogh_y, gogh_y, transb=True)/np.float32(ch*wd**2)

            L1 = np.float32(alpha[l])*F.mean_squared_error(y[l], Variable(mid_orig[l].data))
            L2 = np.float32(beta[l])*F.mean_squared_error(gogh_matrix, Variable(style_mats[l].data))/np.float32(4)
            L += L1+L2

            if i%100==0:
                print i,l,L1.data,L2.data

        L.backward()
        xg += x.grad
        optimizer.update()

        tmp_shape = img_gen.shape
        if args.gpu >= 0:
            img_gen += Clip().forward(img_gen).reshape(tmp_shape) - img_gen
        else:
            def clip(x):
                return -100 if x<-100 else (100 if x>100 else x)
            img_gen += np.vectorize(clip)(img_gen).reshape(tmp_shape) - img_gen

        if i%50==0:
            save_image(img_gen, W, nw, nh, i)





parser = argparse.ArgumentParser(
    description='A Neural Algorithm of Artistic Style')
parser.add_argument('--model', '-m', default='nin_imagenet.caffemodel',
                    help='model file')
parser.add_argument('--orig_img', '-i', default='orig.png',
                    help='Original image')
parser.add_argument('--style_img', '-s', default='style.png',
                    help='Style image')
parser.add_argument('--out_dir', '-o', default='output',
                    help='Output directory')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--iter', default=2000, type=int,
                    help='number of iteration')
parser.add_argument('--lr', default=1.0, type=float,
                    help='learning rate')
parser.add_argument('--lam', default=0.1, type=float,
                    help='original image weight / style weight ratio')
parser.add_argument('--width', '-w', default=435, type=int,
                    help='image width, height')
args = parser.parse_args()

try:
    os.mkdir(args.out_dir)
except:
    pass

if args.gpu >= 0:
	cuda.check_cuda_available()
	cuda.get_device(args.gpu).use()
   	xp = cuda.cupy
else:
   	xp = np


chainer.Function.type_check_enable = False
print "load model... %s"%args.model
func = caffe.CaffeFunction(args.model)
model = func.fs
if args.gpu>=0:
	model.to_gpu()

W = args.width
img_gogh,_,_ = image_resize(args.style_img, W)
img_hongo,nw,nh = image_resize(args.orig_img, W)

generate_image(img_hongo, img_gogh, W, nw, nh, img_gen=None, max_iter=args.iter, lr=args.lr, alpha=[args.lam * x for x in [0.01,0.01,1,1]], beta=[1,1,1,1])

