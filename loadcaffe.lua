local ffi = require 'ffi'
local C = loadcaffe.C


loadcaffe.load = function(prototxt_name, binary_name, cuda_package)
  local cuda_package = cuda_package or 'nn'
  local handle = ffi.new('void*[1]')
  local lua_name = 'file.lua'
  C['loadBinary'](handle, prototxt_name, binary_name)
  C['convertProtoToLua'](handle, lua_name, cuda_package)
  dofile(lua_name)
  local net = nn.Sequential()
  local list_modules = model
  for i,item in ipairs(list_modules) do
    item[2]:cuda()
    if item[2].weight then
      local w = torch.FloatTensor()
      local bias = torch.FloatTensor()
      C['loadModule'](handle, item[1], w:cdata(), bias:cdata())
      w = w:transpose(1,4):transpose(1,3):transpose(1,2)
      item[2].weight:copy(w)
      item[2].bias:copy(bias)
    end
    net:add(item[2])
  end
  C['destroyBinary'](handle)
  print(net)
  return net
end
