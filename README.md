loadcaffe
=========

Load Caffe networks in Torch7

There is no Caffe dependency, only protobuf has to be installed.

Work in progress! For now only imagenet with cuda-convnet2 tested. To load it do:

```lua
require 'loadcaffe'

prototxt_name = 'deploy.prototxt'
binary_name = 'bvlc_reference_caffenet.caffemodel'

model = loadcaffe.load(prototxt_name, binary_name, 'ccn2')
```

https://github.com/BVLC/caffe/wiki/Model-Zoo in plans.
