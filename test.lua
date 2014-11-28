dofile 'loadcaffe.lua'

loadcaffe.convertProtoToLua('/opt/caffe/models/bvlc_reference_caffenet/deploy.prototxt', 'file.lua', 'ccn2')
