# torchonnx

torchonnx is a tool to convert an ONNX model (.onnx file) to a pytorch model (.pth file for model parameters and .py file for neural network structure).

## Installation

Only need to install `pytorch` and `onnx` packages.

## Usage

There is an example in folder `test`.

## Current Supported Operations

- Gemm: torch.nn.Linear
- Convolution: torch.nn.Conv2d
- ReLU: torch.nn.ReLU
- Add: +
