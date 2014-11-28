include 'init.lua'
local C = loadcaffe.C

loadcaffe.convertProtoToLua = function(prototxt_name, lua_name, cuda_package)
  C['convertProtoToLua'](prototxt_name, lua_name, cuda_package)
end


loadcaffe.loadModule = function(list_modules)
  for item in pairs(list_modules) do
    if item.weight then
      local w = torch.FloatTensor()
      local bias = torch.FloatTensor()
      C[''](item[1], w:cdata(), bias:cdata())
      item.weight:copy(w)
      item.bias:copy(bias)
    end
  end
end
