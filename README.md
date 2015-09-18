# chainer-gogh

## Implementation of "A neural algorithm of Artistic style" (http://arxiv.org/abs/1508.06576)
## 解説記事: https://research.preferred.jp/2015/09/chainer-gogh/

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

(VGG, lam=0.0075, after 5000 iteration)

## Usage:
### Chainerをインストール
```
pip install chainer
```
詳しくはhttps://github.com/pfnet/chainer

### モデルをダウンロード
* NIN https://gist.github.com/mavenlin/d802a5849de39225bcc6

お手軽。(`-m nin`)
* VGG https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

きれいな絵がかけるがとても重い。(`-m vgg`)

* GoogLeNet https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet

NIN並に軽く、ポテンシャルもあるはずだが、最適なパラメタがわかっていない。(`-m googlenet`)

* illustration2vec http://illustration2vec.net/   (pre-trained model for tag prediction, version 2.0)

VGGより軽く、二次元画像にとても強いはずだが、最適なパラメタがわかってない。(`-m i2v`)

### CPU実行
```
python chainer-gogh.py -m nin -i input.png -s style.png -o output_dir -g -1
```

### GPU実行
```
python chainer-gogh.py -m nin -i input.png -s style.png -o output_dir -g GPU番号
```

### VGG実行サンプル
```
python chainer-gogh.py -m vgg -i input.png -s style.png -o output_dir -g 0 --width 256
```

### モデルの指定方法
```
-m nin
```
のninを、vgg, googlenet, i2vに切り替えることが可能。
モデルファイルはディレクトリ直下に置いて、デフォルトの名前のまま変えないこと。

### 複数枚同時生成
* まず、input.txtというファイル名で、以下の様なファイルを作る。
```
input0.png style0.png
input1.png style1.png
...
```
そして、chainer-gogh-multi.pyの方を実行
```
python chainer-gogh-multi.py -i input.txt
```
VGGを使うときはGPUのメモリ不足に注意

## パラメタについて
* `--lr`: 学習速度。生成の進捗が遅い時は大きめにする
* `--lam`: これを上げるとinput画像に近くなり、下げるとstyle画像に近くなる
* alpha, beta: 各層から伝播させる誤差にかかる係数。models.pyの中でハードコードされている。

## 注意
* 現在のところ画像は正方形に近いほうがいいです
