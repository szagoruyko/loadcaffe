require 'loadcaffe'

nets_list = torch.load 'nets_list.bin'

for i,netprops in ipairs(nets_list) do
  net = loadcaffe.load(netprops.proto, netprops.binary, 'ccn2')

  input = torch.CudaTensor(32,3,netprops.imsize, netprops.imsize)
  input[{{1,10},{},{},{}}]:copy(netprops.input)

  output = net:forward(input):float()

  r = netprops.output:squeeze()
  g = output[{{1,10},{}}]

  print(netprops.name, (r-g):abs():mean())
end
