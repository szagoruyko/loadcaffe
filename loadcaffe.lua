local C = loadcaffe.C
local ffi = require 'ffi'

loadcaffe.convertProtoToLua = function(prototxt_name, lua_name, cuda_package)
  C['convertProtoToLua'](prototxt_name, lua_name, cuda_package)
end


loadcaffe.loadModules = function(prototxt_name, binary_name, list_modules)
  local handle = ffi.new('void*[1]')
  C['loadBinary'](prototxt_name, binary_name, handle)
  for i,item in ipairs(list_modules) do
    if item[2].weight then
      local w = torch.FloatTensor()
      local bias = torch.FloatTensor()
      C['loadModule'](handle, item[1], w:cdata(), bias:cdata())
      w = w:transpose(1,4):transpose(1,3):transpose(1,2)
      print(#bias)
      print(#item[2].bias)
      item[2].weight:copy(w)
      item[2].bias:copy(bias)
    end
  end
end
