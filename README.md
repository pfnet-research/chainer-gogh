# chainer-gogh

## Implementation of "A neural algorithm of Artistic style" (http://arxiv.org/abs/1508.06576)
## Accompanying article: https://research.preferred.jp/2015/09/chainer-gogh/

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/cat.png" height="150px">


<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_0.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im0.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_1.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im1.png" height="150px">

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_2.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im2.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_3.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im3.png" height="150px">

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_4.jpg" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im4.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_5.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im5.png" height="150px">

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_6.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im6.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_7.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im7.png" height="150px">

(VGG, lam=0.0075, after 5000 iterations)

## Usage:
### Install Chainer
```
pip install chainer
```
See https://github.com/pfnet/chainer for details.

### Download the model
* NIN https://gist.github.com/mavenlin/d802a5849de39225bcc6

お手軽。(`-m nin`)
* VGG https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

きれいな絵がかけるがとても重い。(`-m vgg`, `-m vgg_chainer`)
vgg_chainerではモデルのダウンロードの必要はなく、初回を除いて非常に高速でロードできるようになります(chainer 1.19以降で動作)。

* GoogLeNet https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet

NIN並に軽く、ポテンシャルもあるはずだが、最適なパラメタがわかっていない。(`-m googlenet`)

* illustration2vec http://illustration2vec.net/   (pre-trained model for tag prediction, version 2.0)

VGGより軽く、二次元画像にとても強いはずだが、最適なパラメタがわかってない。(`-m i2v`)

### Run on CPU
```
python chainer-gogh.py -m nin -i input.png -s style.png -o output_dir -g -1
```

### Run on GPU
```
python chainer-gogh.py -m nin -i input.png -s style.png -o output_dir -g GPU番号
```

### Stylize an image with VGG
```
python chainer-gogh.py -m vgg_chainer -i input.png -s style.png -o output_dir -g 0 --width 256
```

### How to specify the model
```
-m nin
```
のninを、vgg, vgg_chainer, googlenet, i2vに切り替えることが可能。
モデルファイルはディレクトリ直下に置いて、デフォルトの名前のまま変えないこと。

### Generate multiple images simultaneously
* First, createa file called input.txt and list the input and output file names:
```
input0.png style0.png
input1.png style1.png
...
```
then, run chainer-gogh-multi.py:
```
python chainer-gogh-multi.py -i input.txt
```
The VGG model uses a lot of GPU memory.

## About the parameters
* `--lr`: learning rate. Increase this when the generation progress is slow.
* `--lam`: increase the make the output image similar to the input, decrease to add more style.
* alpha, beta: coefficients relating to the error propagated from each layer. They are hard coded for each model.

## Advice
* At the moment, using square images (e.g. 32x32) is best.
