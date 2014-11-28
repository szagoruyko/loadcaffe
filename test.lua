require 'loadcaffe'

prototxt_name = '/home/zagoruys/deploy.prototxt'
binary_name = '/home/zagoruys/bvlc_reference_caffenet.caffemodel'

model = loadcaffe.load(prototxt_name, binary_name, 'ccn2')
