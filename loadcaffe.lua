local ffi = require 'ffi'
local C = loadcaffe.C


loadcaffe.load = function(prototxt_name, binary_name, backend)
  local backend = backend or 'nn'
  local handle = ffi.new('void*[1]')

  -- loads caffe model in memory and keeps handle to it in ffi
  local old_val = handle[1]
  C.loadBinary(handle, prototxt_name, binary_name)
  if old_val == handle[1] then return end

  -- transforms caffe prototxt to torch lua file model description and 
  -- writes to a script file
  local lua_name = prototxt_name..'.lua'
  C.convertProtoToLua(handle, lua_name, backend)

  -- executes the script
  local model_definition = io.open(lua_name):read'*all'
  if (model_definition:find'inn%.') then
    require 'inn'
  end
  local model = dofile(lua_name)

  -- goes over the list, copying weights from caffe blobs to torch tensor
  local net = nn.Sequential()
  local list_modules = model
  for i,item in ipairs(list_modules) do
    item[2].name = item[1]
    if item[2].weight then
      local w = torch.FloatTensor()
      local bias = torch.FloatTensor()
      C.loadModule(handle, item[1], w:cdata(), bias:cdata())
      if backend == 'ccn2' then
        w = w:permute(2,3,4,1)
      end
      item[2].weight:copy(w)
      item[2].bias:copy(bias)
    end
    net:add(item[2])
  end
  C.destroyBinary(handle)

  if backend == 'cudnn' or backend == 'ccn2' then
    net:cuda()
  end

  return net
end
