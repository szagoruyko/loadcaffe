loadcaffe
=========

Load Caffe networks in Torch7

There is no Caffe dependency, only protobuf has to be installed. In Ubuntu do:

```
sudo apt-get install libprotobuf-dev
```

To load a network do:

```lua
require 'loadcaffe'

model = loadcaffe.load('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'ccn2')
```

Tested with cuda-convnet2:

* bvlc_reference_rcnn_ilsvrc13
* bvlc_reference_caffenet
* bvlc_alexnet
* finetune_flickr_style
* VGG_CNN_S
* VGG_CNN_M
* VGG_CNN_M_2048
* VGG_CNN_M_1024
* VGG_CNN_M_128
* VGG_CNN_F
* Models used by VGG in ILSVRC-2014, both 16 and 19-layer models

For nets without local response normalization nn, cunn and cudnn can be used, however for max-pooling will be used ccn2.

Rights to caffe.proto belong to the University of California.
