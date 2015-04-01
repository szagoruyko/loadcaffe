local ffi = require 'ffi'
local C = loadcaffe.C


loadcaffe.load = function(prototxt_name, binary_name, cuda_package)
  local cuda_package = cuda_package or 'nn'
  local handle = ffi.new('void*[1]')

  -- loads caffe model in memory and keeps handle to it in ffi
  local old_val = handle[1]
  C.loadBinary(handle, prototxt_name, binary_name)
  if old_val == handle[1] then return end

  -- transforms caffe prototxt to torch lua file model description and 
  -- writes to a script file
  local lua_name = prototxt_name..'.lua'
  C.convertProtoToLua(handle, lua_name, cuda_package)

  -- executes the script, defining global 'model' module list
  local model = dofile(lua_name)

  -- goes over the list, copying weights from caffe blobs to torch tensor
  local net = nn.Sequential()
  local list_modules = model
  for i,item in ipairs(list_modules) do
    item[2]:cuda()
    if item[2].weight then
      local w = torch.FloatTensor()
      local bias = torch.FloatTensor()
      C.loadModule(handle, item[1], w:cdata(), bias:cdata())
      if cuda_package == 'ccn2' then
        w = w:transpose(1,4):transpose(1,3):transpose(1,2)
      end
      item[2].weight:copy(w)
      item[2].bias:copy(bias)
    end
    net:add(item[2])
  end
  C.destroyBinary(handle)
  --print(net)
  return net
end
