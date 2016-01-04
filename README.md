loadcaffe
=========

Load Caffe networks in Torch7

There is no Caffe dependency, only protobuf has to be installed. In Ubuntu do:

```
sudo apt-get install libprotobuf-dev protobuf-compiler
```

In OS X:

```
brew install protobuf
```

Then install the package itself:

```
luarocks install loadcaffe
```

Load a network:

```lua
require 'loadcaffe'

model = loadcaffe.load('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'ccn2')
```

Models from Caffe [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo):

| Network  | ccn2 | nn | cudnn |
| ------------- | :-------------: | :-------: | :---: |
| bvlc_alexnet | + | - | + |
| bvlc_reference_caffenet | + | - | + |
| bvlc_reference_rcnn_ilsvrc13 | + | - | + |
| [finetune_flickr_style](https://gist.github.com/sergeyk/034c6ac3865563b69e60) | + | - | + |
| [VGG_CNN_S](https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9)  | +  | + | + |
| [VGG_CNN_M](https://gist.github.com/ksimonyan/f194575702fae63b2829)  | +  | + | + |
| [VGG_CNN_M_2048](https://gist.github.com/ksimonyan/78047f3591446d1d7b91)  | +  | + | + |
| [VGG_CNN_M_1024](https://gist.github.com/ksimonyan/f0f3d010e6d5f0100274)  | +  | + | + |
| [VGG_CNN_M_128](https://gist.github.com/ksimonyan/976847408258292576a1)  | +  | + | + |
| [VGG_CNN_F](https://gist.github.com/ksimonyan/a32c9063ec8e1118221a)  | +  | + | + |
| [VGG ILSVRC-2014 16-layer](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) | + | + | + |
| [VGG ILSVRC-2014 19-layer](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77) | + | + | + |
| [Network-in-Network Imagenet](https://gist.github.com/mavenlin/d802a5849de39225bcc6) | - | + | + |
| [Network-in-Network CIFAR-10](https://gist.github.com/mavenlin/e56253735ef32c3c296d) | - | + | + |
| [VGG16_SalObjSub](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395) | + | + | + |
| [AlexNex_SalObjSub](https://gist.github.com/jimmie33/0585ed9428dc5222981f) | + | - | + | 
| [Binary Hash Codes](https://gist.github.com/kevinlin311tw/266d4150a1db5810398e) | + | - | + |
| [Oxford 102 Flowers](https://gist.github.com/jimgoo/0179e52305ca768a601f) | + | - | + |
| [Age&Gender](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/) | + | + | + |
| MNIST LeNet | - | + | + |

Loading googlenet is supported by https://github.com/soumith/inception.torch

NN support means both CPU and GPU backends.

You can also use Caffe inside Torch with this: https://github.com/szagoruyko/torch-caffe-binding However you can't use both loadcaffe and caffe in one torch session.

An example of using the package is in [examples/mnist_lenet.lua](examples/mnist_lenet.lua). After running script to train lenet model in Caffe you can easily load and test it in Torch7 on CPU and GPU (with 'cuda' as a first arguments)

Rights to caffe.proto belong to the University of California.
