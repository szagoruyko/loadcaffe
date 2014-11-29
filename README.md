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

Other nets as VGG_CNN can be loaded, but doesn't work yet, weights, however, can be used. We're going to fix this soon.

https://github.com/BVLC/caffe/wiki/Model-Zoo in plans.

Rights to caffe.proto belong to University of California.
