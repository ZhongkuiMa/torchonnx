# torchonnx: Convert ONNX Model to PyTorch Model

**torchonnx** is a tool that helps you convert an **ONNX model** (.onnx file) into a **PyTorch model** (.pth file for model parameters and .py file for neural network structure). 

## Why We Do This?

While PyTorch provides the `torch.onnx` module to convert a PyTorch model to an ONNX model, the reverse process—converting an ONNX model to PyTorch—is not straightforward. Here are the reasons why:

- **Too many versions of ONNX**: The ONNX model format evolves across versions, and different versions contain different operations, which can be confusing. Unfortunately, there's little you can do about this. Maybe this tool has to face such case in the future.
- **Inconsistencies between ONNX and PyTorch**: There are many inconsistencies (names, parameters, even logic, etc.) between ONNX and PyTorch models. This makes it difficult to automatically convert the ONNX model back into a PyTorch model. Perhaps PyTorch doesn't officially support this because they don't think it’s necessary. However, for us in the **NNV (Neural Network Verification)** community, it's essential because ONNX has become the unified model format.
- **Unfriendly ONNX**: ONNX doesn't seem to care about the logic behind operations. It just reads and generates code without concern for how the operations fit together.

## What We Do?

I acknowledge that some tools exist to convert ONNX models to PyTorch models, and I appreciate their work. Currently, the most well-known tool is [onnx2pytorch](https://github.com/Talmaj/onnx2pytorch). However, its code is not optimized for performance and may make the conversion process cumbersome. For example, its `forward` method still requires iterating over all ONNX nodes instead of just using the PyTorch model. In other words, it's more of a decorator for the ONNX model than a true PyTorch model.

Additionally, onnx2pytorch suffers from inefficiencies, especially when converting the ONNX model's initializers to PyTorch tensor parameters. There are repeated operations and significant time-wasting.

To solve these issues, I decided to write my own tool for converting ONNX models to PyTorch models. The idea is simple: convert the ONNX model into two files:
1. A `.py` file defining the neural network structure.
2. A `.pth` file saving the model parameters.

This method avoids dealing with the conversion or construction of PyTorch module objects in code and is both **simple** and **efficient**.

## How to Use?

### Installation Guide

There are no significant installation requirements for this tool. All you need is:
- **PyTorch**: `torch`
- **ONNX**: `onnx`

### Current Supported Features

I have implemented the common operations for feedforward neural networks. I think it is enough for most of the cases. If you find some operations are not supported, please let me know.
