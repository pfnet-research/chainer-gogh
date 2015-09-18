
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

from models import *

import pickle

def subtract_mean(x0):
    x = x0.copy()
    x[0,:,:] -= 120
    x[1,:,:] -= 120
    x[2,:,:] -= 120
    return x
def add_mean(x0):
    x = x0.copy()
    x[0,:,:] += 120
    x[1,:,:] += 120
    x[2,:,:] += 120
    return x


def image_resize(img_file, width):
    gogh = Image.open(img_file)
    orig_w, orig_h = gogh.size[0], gogh.size[1]
    if orig_w>orig_h:
        new_w = width
        new_h = width*orig_h/orig_w
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((3,new_h,new_w))
        print("image resized to: ", gogh.shape)
        hoge= np.zeros((3,width,width), dtype=np.float32)
        hoge[:,width-new_h:,:] = gogh[:,:,:]
        gogh = subtract_mean(hoge)
    else:
        new_w = width*orig_w/orig_h
        new_h = width
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((3,new_h,new_w))
        print("image resized to: ", gogh.shape)
        hoge= np.zeros((3,width,width), dtype=np.float32)
        hoge[:,:,width-new_w:] = gogh[:,:,:]
        gogh = subtract_mean(hoge)
    return xp.asarray(gogh), new_w, new_h

def save_image(img, width, new_w, new_h, out_fn):
    def to_img(x):
        im = np.zeros((new_h,new_w,3))
        im[:,:,0] = x[2,:,:]
        im[:,:,1] = x[1,:,:]
        im[:,:,2] = x[0,:,:]
        def clip(a):
            return 0 if a<0 else (255 if a>255 else a)
        im = np.vectorize(clip)(im).astype(np.uint8)
        Image.fromarray(im).save(out_fn)

    if args.gpu>=0:
        img_cpu = add_mean(img.get())
    else:
        img_cpu = add_mean(img)
    if width==new_w:
        to_img(img_cpu[:,width-new_h:,:])
    else:
        to_img(img_cpu[:,:,width-new_w:])


def get_matrix(y):
    ch = y.data.shape[1]
    wd = y.data.shape[2]
    gogh_y = F.reshape(y, (y.data.shape[0],ch,wd**2))
    gogh_matrix = F.batch_matmul(gogh_y, gogh_y, transb=True)/np.float32(ch*wd**2)
    return gogh_matrix



class Clip(chainer.Function):
    def forward(self, x):
        ret = cuda.elementwise(
            'T x','T ret',
            '''
                ret = x<-120?-120:(x>136?136:x);
            ''','clip')(x)
        return ret

def generate_image(img_orig, img_style, width, nw, nh, max_iter, lr, img_gen=None):
    batch_size = img_orig.shape[0]
    mid_orig = nn.forward(Variable(img_orig, volatile=True))
    style_mats = [get_matrix(y) for y in nn.forward(Variable(img_style, volatile=True))]

    if img_gen is None:
        if args.gpu >= 0:
            img_gen_ = xp.random.uniform(-20,20,(3,width,width),dtype=np.float32)
            img_gen = xp.random.uniform(-20,20,(batch_size,3,width,width),dtype=np.float32)
            img_gen[:,:,:,:] = img_gen_
        else:
            img_gen = np.random.uniform(-20,20,(batch_size,3,width,width)).astype(np.float32)
            img_gen = np.random.uniform(-20,20,(batch_size,3,width,width)).astype(np.float32)
            img_gen[:,:,:,:] = img_gen_
    x = Variable(img_gen)
    xg = xp.zeros_like(x.data)
    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup((img_gen,xg))
    for i in range(max_iter):

        x = Variable(img_gen)
        y = nn.forward(x)

        optimizer.zero_grads()
        L = Variable(xp.zeros((), dtype=np.float32))
        for l in range(len(y)):
            gogh_matrix = get_matrix(y[l])

            L1 = np.float32(args.lam) * np.float32(nn.alpha[l])*F.mean_squared_error(y[l], Variable(mid_orig[l].data))
            L2 = np.float32(nn.beta[l])*F.mean_squared_error(gogh_matrix, Variable(style_mats[l].data))/np.float32(len(y))
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
                return -120 if x<-120 else (136 if x>136 else x)
            img_gen += np.vectorize(clip)(img_gen).reshape(tmp_shape) - img_gen

        if i%50==0:
            for j in range(img_gen.shape[0]):
                save_image(img_gen[j], W, nw[j], nh[j], args.out_dir+"_%d/im_%05d.png"%(j,i))

    for j in range(img_gen.shape[0]):
        save_image(img_gen[j], W, nw[j], nh[j], args.out_dir+"_last/im_%d.png"%(j))



parser = argparse.ArgumentParser(
    description='A Neural Algorithm of Artistic Style')
parser.add_argument('--model', '-m', default='nin',
                    help='model file (nin, vgg, i2v, googlenet)')
parser.add_argument('--input_file', '-i', default='input.txt',
                    help='input data text')
parser.add_argument('--out_dir', '-o', default='output',
                    help='Output directory')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--iter', default=5000, type=int,
                    help='number of iteration')
parser.add_argument('--lr', default=4.0, type=float,
                    help='learning rate')
parser.add_argument('--lam', default=0.005, type=float,
                    help='original image weight / style weight ratio')
parser.add_argument('--width', '-w', default=435, type=int,
                    help='image width, height')
args = parser.parse_args()

input_data = open(args.input_file, "r").readlines()

for i in range(len(input_data)):
    try:
        os.mkdir(args.out_dir+"_%d"%i)
    except:
        pass
try:
    os.mkdir(args.out_dir+"_last")
except:
    pass

if args.gpu >= 0:
    cuda.check_cuda_available()
    chainer.Function.type_check_enable = False
    cuda.get_device(args.gpu).use()
    xp = cuda.cupy
else:
    xp = np

if 'nin' in args.model:
    nn = NIN()
elif 'vgg' in args.model:
    nn = VGG()
elif 'i2v' in args.model:
    nn = I2V()
elif 'googlenet' in args.model:
    nn = GoogLeNet()
else:
    print 'invalid model name. you can use (nin, vgg, i2v, googlenet)'
if args.gpu>=0:
	nn.model.to_gpu()

W = args.width
img_orig = xp.zeros((len(input_data), 3, W, W), dtype=np.float32)
img_style = xp.zeros((len(input_data), 3, W, W), dtype=np.float32)
nw = []
nh = []
for i in range(len(input_data)):
    fns = input_data[i].split()
    img_orig[i,:,:,:],w_,h_ = image_resize(fns[0], W)
    img_style[i,:,:,:],_,_ = image_resize(fns[1], W)
    nw.append(w_)
    nh.append(h_)

generate_image(img_orig, img_style, W, nw, nh, img_gen=None, max_iter=args.iter, lr=args.lr)
