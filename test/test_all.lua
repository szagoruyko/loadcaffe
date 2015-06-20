require 'loadcaffe'

nets_list = torch.load 'nets_list.bin'

j = tonumber(arg[1] or -1)
dtype = arg[2] or 'ccn2'

if j ~= -1 then
  nets_list2 = {}
  nets_list2[1] = nets_list[j]
  nets_list = nets_list2
end

batch_size = dtype == 'ccn2' and 32 or 10

for i,netprops in ipairs(nets_list) do
    net = loadcaffe.load(netprops.proto, netprops.binary, dtype)
    net:evaluate()

    input = torch.CudaTensor(batch_size,3,netprops.imsize, netprops.imsize)
    input[{{1,10},{},{},{}}]:copy(netprops.input)

    output = net:forward(input):float()

    r = netprops.output:squeeze()
    g = output[{{1,10},{}}]

    print(netprops.name, dtype, (r-g):abs():mean())
    collectgarbage()
end
