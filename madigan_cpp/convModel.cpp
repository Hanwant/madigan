#include <list>
#include <torch/torch.h>
#include "Synth.h"

struct ConvNet: torch::nn::Module{

  ConvNet(int minTf, int nAssets, int actionAtoms, int dModel, std::vector<int> channels={32, 64, 64},
          std::vector<int> kernels = {3, 3, 3}, std::vector<int> strides = {1, 1, 1}):
    minTf(minTf), nAssets(nAssets), actionAtoms(actionAtoms), dModel(dModel), channels(channels),
    kernels(kernels), strides(strides){
    c1 = register_module("c1", torch::nn::Conv1d(torch::nn::Conv1dOptions(nAssets, channels[0], kernels[0]
                                                                          ).stride(strides[0])));
    c2 = register_module("c2", torch::nn::Conv1d(torch::nn::Conv1dOptions(channels[0], channels[1], kernels[1]).stride(strides[1])));
    c3 = register_module("c3", torch::nn::Conv1d(torch::nn::Conv1dOptions(channels[1], channels[2], kernels[2]
                                                                          ).stride(strides[2])));
    torch::Tensor input = torch::rand({1, nAssets, minTf});
    torch::Tensor exampleOut = c3(c2(c1(input)));
    fc = register_module("fc", torch::nn::Linear(exampleOut.size(1) * exampleOut.size(2), dModel));
    out = register_module("out", torch::nn::Linear(dModel, nAssets*actionAtoms));

  }

  torch::Tensor forward(torch::Tensor input){
    input = act(c1(input));
    input = act(c2(input));
    input = act(c3(input));
    input = input.view({input.size(0), -1});
    input = act(fc(input));
    input = out(input);
    return input.view({input.size(0), nAssets, actionAtoms});
  }


  int minTf;
  int nAssets;
  int actionAtoms;
  int dModel;
  std::vector<int> channels;
  std::vector<int> kernels;
  std::vector<int> strides;

  torch::nn::Conv1d c1=nullptr;
  torch::nn::Conv1d c2=nullptr;
  torch::nn::Conv1d c3=nullptr;
  torch::nn::Linear fc=nullptr;
  torch::nn::Linear out=nullptr;
  torch::nn::ReLU act;
};

int main(){

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available \nUsing GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  Synth env = Synth(4, 64, 1'000'000, 11);
  ConvNet net = ConvNet(64, 4, 12, 256);
  net.to(device);

  std::cout<<"Net Initialized" << std::endl;

  torch::Tensor input = torch::randn({1, net.nAssets, net.minTf}).to(device);
  auto output = net.forward(input);
  std::cout << "Out size: " << output.size(0) << " " << output.size(1) << " " << output.size(2) << std::endl;
  std::cout << output << std::endl;

  return 0;


}


