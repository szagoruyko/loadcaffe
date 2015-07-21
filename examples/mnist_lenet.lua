require 'loadcaffe'
require 'xlua'
require 'optim'
mnist = require 'mnist'

-- to train lenet network please follow the steps
-- provided in CAFFE_DIR/examples/mnist
prototxt = '/opt/caffe/examples/mnist/lenet.prototxt'
binary = '/opt/caffe/examples/mnist/lenet_iter_10000.caffemodel'

-- this will load the network and print it's structure
net = loadcaffe.load(prototxt, binary)

-- load test data
testData = mnist.testdataset()

-- preprocess by dividing by 256
images = testData.data:float():div(256)

if arg[1] == 'cuda' then
  net:cuda()
  images = images:cuda()
else
  net:float()
end

-- will be used to print the results
confusion = optim.ConfusionMatrix(10)

for i=1,images:size(1) do
  _,y = net:forward(images[i]:view(1,28,28)):max(1)
  confusion:add(y[1], testData.label[i]+1)
end

-- that's all! will print the error and confusion matrix
print(confusion)
