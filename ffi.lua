local ffi = require 'ffi'

ffi.cdef[[
void convertProtoToLua(const char* prototxt_name, const char* lua_name, const char* cuda_package);
]]

--loadcaffe.C = ffi.load(package.searchpath('libloadcaffe', package.cpath))
loadcaffe.C = ffi.load('./build/libloadcaffe.so')
