//
//  main.cpp
//  loadcaffe
//
//  Created by Sergey Zagoruyko on 28/11/14.
//  Copyright (c) 2014 Sergey Zagoruyko. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

#include <TH/TH.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "build/caffe.pb.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;


extern "C" {
void loadBinary(void** handle, const char* prototxt_name, const char* binary_name);
void convertProtoToLua(void** handle, const char* lua_name, const char* cuda_package);
void loadModule(const void** handle, const char* name, THFloatTensor* weight, THFloatTensor* bias);
void destroyBinary(void** handle);
}


bool ReadProtoFromTextFile(const char* filename, Message* proto) {
    int fd = open(filename, O_RDONLY);
    if(fd == 0)
      return false;
    
    FileInputStream* input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
}


bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
    int fd = open(filename, O_RDONLY);
    if(fd == 0)
      return false;
    
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(1073741824, 536870912);
    
    bool success = proto->ParseFromCodedStream(coded_input);
    
    delete coded_input;
    delete raw_input;
    close(fd);
    return success;
}


void convertProtoToLua(void** handle, const char* lua_name, const char* cuda_package)
{
  const caffe::NetParameter netparam = *(const caffe::NetParameter*)handle[1];

  std::ofstream ofs (lua_name);

  ofs << "require '" << cuda_package << "'\n";
  ofs << "require 'cunn'\n";
  ofs << "model = {}\n";
  
  int num_output = netparam.input_dim_size();
    for (int i=0; i<netparam.layers_size(); ++i)
    {
	char buf[1024];
	bool success = true;
        bool insert_view = false;
        switch(netparam.layers(i).type())
        {
            case caffe::LayerParameter::CONVOLUTION:
            {
                auto &param = netparam.layers(i).convolution_param();
                int groups = param.group() == 0 ? 1 : param.group();
                int nInputPlane = netparam.layers(i).blobs(0).channels()*groups;
                int nOutputPlane = param.num_output();
                num_output = nOutputPlane;
                int kW = param.kernel_w();
                int kH = param.kernel_h();
                int dW = param.stride_w();
                int dH = param.stride_h();
                if(kW==0 || kH==0)
                {
                  kW = param.kernel_size();
                  kH = kW;
                }
                if(dW==0 || dH==0)
                {
                  dW = param.stride();
                  dH = dW;
                }
                int padding = param.pad();
                if(std::string(cuda_package) == "ccn2")
                  sprintf(buf, "ccn2.SpatialConvolution(%d, %d, %d, %d, %d, %d)", nInputPlane, nOutputPlane, kW, dW, padding, groups);
                else
                  sprintf(buf, "%s.SpatialConvolution%s(%d, %d, %d, %d, %d, %d, %d)", cuda_package, std::string(cuda_package)=="nn" ? "MM" : "", nInputPlane, nOutputPlane, kW, kH, dW, dH, padding);
                break;
            }
            case caffe::LayerParameter::POOLING:
            {
                auto &param = netparam.layers(i).pooling_param();
                std::string ptype = param.pool() == caffe::PoolingParameter::MAX ? "Max" : "Avg";
                int kW = param.kernel_w();
                int kH = param.kernel_h();
                int dW = param.stride_w();
                int dH = param.stride_h();
                if(kW==0 || kH==0)
                {
                  kW = param.kernel_size();
                  kH = kW;
                }
                if(dW==0 || dH==0)
                {
                  dW = param.stride()==0;
                  dH = dW;
                }
                if(std::string(cuda_package) == "ccn2")
                  sprintf(buf, "ccn2.Spatial%sPooling(%d, %d)", ptype.c_str(), kW, dW);
                else
                  sprintf(buf, "%s.Spatial%sPooling(%d, %d, %d, %d)", cuda_package, ptype=="Avg" ? "Average" : "Max", kW, kH, dW, dH);
                break;
            }
            case caffe::LayerParameter::RELU:
            {
	        sprintf(buf, "nn.ReLU()");
                break;
            }
            case caffe::LayerParameter::LRN:
            {
              if(std::string(cuda_package) == "ccn2")
              {
                auto &param = netparam.layers(i).lrn_param();
                int local_size = param.local_size();
                float alpha = param.alpha();
                float beta = param.beta();
                sprintf(buf, "ccn2.SpatialCrossResponseNormalization(%d, %.6f, %.4f)", local_size, alpha, beta);
              }
              else
              {
                success = false;
              }
              break;
            }
            case caffe::LayerParameter::INNER_PRODUCT:
            {
                auto &param = netparam.layers(i).inner_product_param();
                int nInputPlane = netparam.layers(i).blobs(0).width();
                int nOutputPlane = param.num_output();
                insert_view = num_output != nInputPlane;
                num_output = nOutputPlane;
                sprintf(buf, "nn.Linear(%d, %d)", nInputPlane, nOutputPlane);
                break;
            }
            case caffe::LayerParameter::DROPOUT:
            {
                sprintf(buf, "nn.Dropout(%f)", netparam.layers(i).dropout_param().dropout_ratio());
                break;
            }
            case caffe::LayerParameter::SOFTMAX_LOSS:
            {
	        sprintf(buf, "nn.SoftMax()");
                break;
            }
            default:
            {
                std::cout << "MODULE " << netparam.layers(i).type() << " UNDEFINED\n";
		success = false;
            }
        }

        if(insert_view)
        {
          if(std::string(cuda_package) == "ccn2")
            ofs << "table.insert(model, {'reshape', nn.Transpose({4,1},{4,2},{4,3})})\n";
          ofs << "table.insert(model, {'view', nn.View(-1):setNumInputDims(3)})\n";
        }
	if(success)
	  ofs << "table.insert(model, {\"" << netparam.layers(i).name() << "\", " << buf << "})\n";
	else
	  ofs << "-- module \"" << netparam.layers(i).name() << "\" not found\n";
    }
}


void loadBinary(void** handle, const char* prototxt_name, const char* binary_name)
{
  caffe::NetParameter* netparam = (caffe::NetParameter*)handle;
  netparam = new caffe::NetParameter();
  ReadProtoFromTextFile(prototxt_name, netparam);
  bool success = ReadProtoFromBinaryFile(binary_name, netparam);
  if(success)
    std::cout << "Successfully loaded " << binary_name << std::endl;
  else
    std::cout << "Couldn't load " << binary_name << std::endl;

  handle[1] = netparam;
}

void destroyBinary(void** handle)
{
  const caffe::NetParameter* netparam2 = (const caffe::NetParameter*)handle[1];
  delete netparam2;
}

void loadModule(const void** handle, const char* name, THFloatTensor* weight, THFloatTensor* bias)
{
  if(handle == NULL)
  {
    std::cout << "network not loaded!\n";
    return;
  }

  const caffe::NetParameter* netparam = (const caffe::NetParameter*)handle[1];

  int n = netparam->layers_size();
  for(int i=0; i<n; ++i)
  {
    auto &layer = netparam->layers(i);
    if(std::string(name) == layer.name())
    {
      int nInputPlane = layer.blobs(0).channels();
      int nOutputPlane = layer.blobs(0).num();
      int kW = layer.blobs(0).width();
      int kH = layer.blobs(0).height();
      printf("%s: %d %d %d %d\n", name, nOutputPlane, nInputPlane, kW, kH);
      
      THFloatTensor_resize4d(weight, nOutputPlane, nInputPlane, kW, kH);
      memcpy(THFloatTensor_data(weight), layer.blobs(0).data().data(), sizeof(float)*nOutputPlane*nInputPlane*kW*kH);

      THFloatTensor_resize1d(bias, layer.blobs(1).data_size());
      memcpy(THFloatTensor_data(bias), layer.blobs(1).data().data(), sizeof(float)*layer.blobs(1).data_size());
    }
  }
}
