# torchonnx: Convert ONNX Model to PyTorch Model 🔥

**torchonnx** is an amazing tool that lets you easily convert an **ONNX model** (.onnx file) into a **PyTorch model** (.pth file for model parameters and .py file for neural network structure). ⚡

## Why We Do This? 🤔

While PyTorch provides the `torch.onnx` module to convert a PyTorch model to an ONNX model, the reverse process—converting an ONNX model to PyTorch—is not straightforward. So, why are we doing this? Here's why! 💥

- **Too many versions of ONNX** 😖: The ONNX model format evolves across versions, and different versions have different operations. It’s messy, confusing, and frustrating! 😤 Unfortunately, we might have to deal with this in the future... but we'll face it head-on! 💪
- **Inconsistencies between ONNX and PyTorch** ⚡: There are tons of inconsistencies between ONNX and PyTorch models (names, parameters, even logic 🤯)! This makes converting ONNX back into PyTorch tricky. PyTorch doesn’t officially support this conversion, probably because they think it’s unnecessary. But for us in the **NNV (Neural Network Verification)** community, it’s essential! ONNX has become the **unified model format**! 🌍
- **Unfriendly ONNX** 😩: ONNX doesn’t care about how operations should logically fit together! It just generates code without understanding the real flow. We need something better! 💥

## What We Do? 🔥

I know there are some tools out there to convert ONNX models to PyTorch models, and I appreciate their efforts 🙌. But let’s be real—most of them fall short in terms of performance and ease of use. The most well-known tool, [onnx2pytorch](https://github.com/Talmaj/onnx2pytorch), is great but... its code isn’t optimized for performance, and it often makes the process unnecessarily complicated. 😵

For example, the `forward` method still iterates over **all ONNX nodes** instead of just using the PyTorch model! 😱 It’s more of a **decorator** for the ONNX model than a true PyTorch model. And don’t even get me started on the inefficiencies when converting the ONNX model’s initializers to PyTorch tensor parameters. 🙄

So, I decided to take matters into my own hands and build a tool that converts ONNX models to PyTorch models **efficiently** and **effectively**. 💥

Here’s the simple and powerful idea: we’ll convert the ONNX model into **two files**:

1. A `.py` file defining the neural network structure.
2. A `.pth` file saving the model parameters.

This means no hassle with constructing PyTorch module objects in code. It’s **simple**, **clean**, and **super-efficient**! 🚀

## How to Use? 🔧

### Installation Guide 🛠️

Good news—there are **no complicated installation steps**! 🎉 All you need is:

- **PyTorch**: `torch` ✅
- **ONNX**: `onnx` ✅

If you haven’t installed them yet, just refer to how to install [PyTorch](https://pytorch.org/) and [ONNX](https://github.com/onnx/onnx) on their official websites. 🌐

### Example Usage 📚

There is an example about ViT model in `test` folder ([ViT benchmark](https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/vit/onnx) from [VNNCOMP'23](https://sites.google.com/view/vnn2023/home)). Note that you need to use [slimonnx](https://github.com/ZhongkuiMa/slimonnx) to simplify the model first and take its simplified version to generate the pytorch code because there are some unsupported operations in the original model.

```python
from torchonnx import TorchONNX

if __name__ == "__main__":
    # The following
    file_path = "../nets/ibp_3_3_8_v22_simplified.onnx"
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
```

## Current Supported Features 🌟

I have implemented most of commonly used operations in feedforward neural networks. Transformer-based architectures will be treated as several basic operations.