# chainer-gogh

## Implementation of "A neural algorithm of Artistic style" (http://arxiv.org/abs/1508.06576)

- pip install chainer
- download network-in-network caffemodel from  https://gist.github.com/mavenlin/d802a5849de39225bcc6  (wget https://www.dropbox.com/s/0cidxafrb2wuwxw/nin_imagenet.caffemodel?dl=1 -O nin_imagenet.caffemodel)
- python chainer-gogh.py -i t1.png -s katsu.png -o output.png -g 0
