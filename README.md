# TorchONNX: Convert ONNX Model to PyTorch Model

**torchonnx** is an amazing tool that lets you easily convert an **ONNX model** (.onnx file) into a **PyTorch model** (.pth file for model parameters and .py file for neural network structure). ⚡

## Why We Do This? 🤔

While PyTorch provides the `torch.onnx` module to convert a PyTorch model to an ONNX model, the reverse process—converting an ONNX model to PyTorch—is not straightforward. So, why are we doing this? Here's why! 💥

- **Too many versions of ONNX** 😖: The ONNX model format evolves across versions, and different versions have different operations. It’s messy, confusing, and frustrating! 😤 Unfortunately, we might have to deal with this in the future... but we'll face it head-on! 💪
- **Inconsistencies between ONNX and PyTorch** ⚡: There are tons of inconsistencies between ONNX and PyTorch models (names, parameters, even logic 🤯)! This makes converting ONNX back into PyTorch tricky. PyTorch doesn’t officially support this conversion, probably because they think it’s unnecessary. But for us in the **NNV (Neural Network Verification)** community, it’s essential! ONNX has become the **unified model format**! 🌍
- **Unfriendly ONNX** 😩: ONNX doesn’t care about how operations should logically fit together! It just generates code without understanding the real flow. We need something better! 💥

## What We Do? 🔥

> We know there are some tools out there to convert ONNX models to PyTorch models, and we appreciate their efforts 🙌. But let’s be real — most of them fall short in terms of performance and ease of use. The most well-known tool, [onnx2pytorch](https://github.com/Talmaj/onnx2pytorch), is great but... its code isn’t optimized for performance, and it often makes the process unnecessarily complicated.
> For example, the `forward` method still iterates over **all ONNX nodes** instead of just using the PyTorch model! It’s more of a **decorator** for the ONNX model than a true PyTorch model. And don’t even get me started on the inefficiencies when converting the ONNX model’s initializers to PyTorch tensor parameters.

But we don't expect to do that thing! We want to make a "complier" that converts ONNX models to a PyTorch module file. We want to make it **simple**, **clean**, and **efficient**! 🚀

## Features

So, we decided to take matters into my own hands and build a tool that converts ONNX models to PyTorch models **efficiently** and **effectively**. 💥

Here’s the simple and powerful idea: we’ll convert the ONNX model into **two files**:

1. A .py file defining the neural network structure.
2. A .pth file saving the model parameters.

This means no hassle with constructing PyTorch module objects in code. It’s **simple**, **clean**, and **super-efficient**! 🚀

More, this tool is **pure Python**, **structured**, and **easy to read**. It’s designed for designers and researchers who have the same goal as me, so it aims to be **easily extensible**. You can add new operations or modify existing ones without breaking the code. It’s all about **flexibility** and **adaptability**! 🌈

### Current Supported Features 🌟

We have implemented most of commonly used operations in feedforward neural networks. Transformer-based architectures will be treated as several basic operations.

## How to Use? 🔧

### Installation Guide 🛠️

Good news—there are **no complicated installation steps**! 🎉 All you need is Python>=3.10 (we are using Python 3.12) with the following libraries✅:

- `onnx=1.17.0`
- `numpy=2.2.4`

If you haven’t installed them yet, just refer to how to install [PyTorch](https://pytorch.org/) and [ONNX](https://github.com/onnx/onnx) on their official websites. 🌐

### Example Usage 📚

#### Test Examples of VNNCOMP'24

You need to get the repo of [vnncomp2024](https://github.com/ChristopherBrix/vnncomp2024_benchmarks). This repo does not contain the benchmarks folder because it is about 20GB. The testing examples are in the `test_vnncomp` folder. Then you make sure the following folder structure:

```
torchonnx/
│   ├── torchonnx/
│   ├── README.md
│   └── test_vnncomp/
└── ...
vnncomp2024/
│   ├── benchmarks/
└── ...
```

#### An Example of ViT Model

There is an example about ViT model in `test` folder ([ViT benchmark](https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/vit/onnx) from [VNNCOMP'23](https://sites.google.com/view/vnn2023/home)). Note that you need to use [slimonnx](https://github.com/ZhongkuiMa/slimonnx) to simplify the model first and take its simplified version to generate the pytorch code because there are some unsupported operations in the original model.

[netron.app](netron.app) is a good way to check the computation graph of the onnx file.

```python
from torchonnx import TorchONNX

if __name__ == "__main__":
    # The following
    file_path = "../nets/ibp_3_3_8_v22_simplified.onnx"
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
```

The following is an example of generated pytorch code:

```python
"""
This file was generated by the torchconverter.
"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F


# Model name: torch_jit_slimmed
# Input name: input_0
# Input shape: [1, 3, 32, 32]
# Output name: output_71
# Output shape: [1, 10]

class Ibp338V22Simplified(Module):

    def __init__(self):
        super(Module, self).__init__()

        self.data = torch.load('../nets/ibp_3_3_8_v22_simplified.pth')

        self.Conv_1 = nn.Conv2d(3, 48, (8, 8), stride=8)
        self.Conv_1.weight.data = self.data['Initializer_1']
        self.Conv_1.bias.data = self.data['Initializer_2']

        self.Gemm_6 = nn.Linear(48, 48)
        self.Gemm_6.weight.data = self.data['Initializer_39']
        self.Gemm_6.bias.data = self.data['Initializer_40']

        self.Gemm_9 = nn.Linear(48, 48)
        self.Gemm_9.weight.data = self.data['Initializer_37']
        self.Gemm_9.bias.data = self.data['Initializer_38']

        self.Gemm_12 = nn.Linear(48, 48)
        self.Gemm_12.weight.data = self.data['Initializer_35']
        self.Gemm_12.bias.data = self.data['Initializer_36']

        self.Softmax_17 = nn.Softmax(dim=-1)
        self.Gemm_21 = nn.Linear(48, 48)
        self.Gemm_21.weight.data = self.data['Initializer_9']
        self.Gemm_21.bias.data = self.data['Initializer_3']

        self.Gemm_23 = nn.Linear(48, 96)
        self.Gemm_23.weight.data = self.data['Initializer_41']
        self.Gemm_23.bias.data = self.data['Initializer_42']

        self.Relu_24 = nn.ReLU()
        self.Gemm_25 = nn.Linear(96, 48)
        self.Gemm_25.weight.data = self.data['Initializer_10']
        self.Gemm_25.bias.data = self.data['Initializer_4']

        self.Gemm_27 = nn.Linear(48, 48)
        self.Gemm_27.weight.data = self.data['Initializer_47']
        self.Gemm_27.bias.data = self.data['Initializer_48']

        self.Gemm_30 = nn.Linear(48, 48)
        self.Gemm_30.weight.data = self.data['Initializer_45']
        self.Gemm_30.bias.data = self.data['Initializer_46']

        self.Gemm_33 = nn.Linear(48, 48)
        self.Gemm_33.weight.data = self.data['Initializer_43']
        self.Gemm_33.bias.data = self.data['Initializer_44']

        self.Softmax_38 = nn.Softmax(dim=-1)
        self.Gemm_42 = nn.Linear(48, 48)
        self.Gemm_42.weight.data = self.data['Initializer_11']
        self.Gemm_42.bias.data = self.data['Initializer_5']

        self.Gemm_44 = nn.Linear(48, 96)
        self.Gemm_44.weight.data = self.data['Initializer_49']
        self.Gemm_44.bias.data = self.data['Initializer_50']

        self.Relu_45 = nn.ReLU()
        self.Gemm_46 = nn.Linear(96, 48)
        self.Gemm_46.weight.data = self.data['Initializer_12']
        self.Gemm_46.bias.data = self.data['Initializer_6']

        self.Gemm_48 = nn.Linear(48, 48)
        self.Gemm_48.weight.data = self.data['Initializer_55']
        self.Gemm_48.bias.data = self.data['Initializer_56']

        self.Gemm_51 = nn.Linear(48, 48)
        self.Gemm_51.weight.data = self.data['Initializer_53']
        self.Gemm_51.bias.data = self.data['Initializer_54']

        self.Gemm_54 = nn.Linear(48, 48)
        self.Gemm_54.weight.data = self.data['Initializer_51']
        self.Gemm_54.bias.data = self.data['Initializer_52']

        self.Softmax_59 = nn.Softmax(dim=-1)
        self.Gemm_63 = nn.Linear(48, 48)
        self.Gemm_63.weight.data = self.data['Initializer_13']
        self.Gemm_63.bias.data = self.data['Initializer_7']

        self.Gemm_65 = nn.Linear(48, 96)
        self.Gemm_65.weight.data = self.data['Initializer_57']
        self.Gemm_65.bias.data = self.data['Initializer_58']

        self.Relu_66 = nn.ReLU()
        self.Gemm_67 = nn.Linear(96, 48)
        self.Gemm_67.weight.data = self.data['Initializer_14']
        self.Gemm_67.bias.data = self.data['Initializer_8']

        self.Gemm_70 = nn.Linear(48, 10)
        self.Gemm_70.weight.data = self.data['Initializer_33']
        self.Gemm_70.bias.data = self.data['Initializer_34']

    def forward(self, input_0: Tensor) -> Tensor:
        Conv_1_0 = self.Conv_1(input_0)
        Reshape_2_0 = Conv_1_0.reshape((1, 48, 16))
        Transpose_3_0 = Reshape_2_0.permute((0, 2, 1))
        Concat_4_0 = torch.cat([self.data['Initializer_20'], Transpose_3_0], dim=1)
        Add_5_0 = Concat_4_0 + self.data['Initializer_0']
        Gemm_6_0 = self.Gemm_6(Add_5_0)
        Reshape_7_0 = Gemm_6_0.reshape((1, 17, 3, 16))
        Transpose_8_0 = Reshape_7_0.permute((0, 2, 1, 3))
        Gemm_9_0 = self.Gemm_9(Add_5_0)
        Reshape_10_0 = Gemm_9_0.reshape((1, 17, 3, 16))
        Transpose_11_0 = Reshape_10_0.permute((0, 2, 3, 1))
        Gemm_12_0 = self.Gemm_12(Add_5_0)
        Reshape_13_0 = Gemm_12_0.reshape((1, 17, 3, 16))
        Transpose_14_0 = Reshape_13_0.permute((0, 2, 1, 3))
        MatMul_15_0 = Transpose_14_0 @ Transpose_11_0
        Mul_16_0 = MatMul_15_0 * self.data['Initializer_15']
        Softmax_17_0 = self.Softmax_17(Mul_16_0)
        MatMul_18_0 = Softmax_17_0 @ Transpose_8_0
        Transpose_19_0 = MatMul_18_0.permute((0, 2, 1, 3))
        Reshape_20_0 = Transpose_19_0.reshape((1, 17, 48))
        Gemm_21_0 = self.Gemm_21(Reshape_20_0)
        Add_22_0 = Gemm_21_0 + Add_5_0
        Gemm_23_0 = self.Gemm_23(Add_22_0)
        Relu_24_0 = self.Relu_24(Gemm_23_0)
        Gemm_25_0 = self.Gemm_25(Relu_24_0)
        Add_26_0 = Gemm_25_0 + Add_22_0
        Gemm_27_0 = self.Gemm_27(Add_26_0)
        Reshape_28_0 = Gemm_27_0.reshape((1, 17, 3, 16))
        Transpose_29_0 = Reshape_28_0.permute((0, 2, 1, 3))
        Gemm_30_0 = self.Gemm_30(Add_26_0)
        Reshape_31_0 = Gemm_30_0.reshape((1, 17, 3, 16))
        Transpose_32_0 = Reshape_31_0.permute((0, 2, 3, 1))
        Gemm_33_0 = self.Gemm_33(Add_26_0)
        Reshape_34_0 = Gemm_33_0.reshape((1, 17, 3, 16))
        Transpose_35_0 = Reshape_34_0.permute((0, 2, 1, 3))
        MatMul_36_0 = Transpose_35_0 @ Transpose_32_0
        Mul_37_0 = MatMul_36_0 * self.data['Initializer_16']
        Softmax_38_0 = self.Softmax_38(Mul_37_0)
        MatMul_39_0 = Softmax_38_0 @ Transpose_29_0
        Transpose_40_0 = MatMul_39_0.permute((0, 2, 1, 3))
        Reshape_41_0 = Transpose_40_0.reshape((1, 17, 48))
        Gemm_42_0 = self.Gemm_42(Reshape_41_0)
        Add_43_0 = Gemm_42_0 + Add_26_0
        Gemm_44_0 = self.Gemm_44(Add_43_0)
        Relu_45_0 = self.Relu_45(Gemm_44_0)
        Gemm_46_0 = self.Gemm_46(Relu_45_0)
        Add_47_0 = Gemm_46_0 + Add_43_0
        Gemm_48_0 = self.Gemm_48(Add_47_0)
        Reshape_49_0 = Gemm_48_0.reshape((1, 17, 3, 16))
        Transpose_50_0 = Reshape_49_0.permute((0, 2, 1, 3))
        Gemm_51_0 = self.Gemm_51(Add_47_0)
        Reshape_52_0 = Gemm_51_0.reshape((1, 17, 3, 16))
        Transpose_53_0 = Reshape_52_0.permute((0, 2, 3, 1))
        Gemm_54_0 = self.Gemm_54(Add_47_0)
        Reshape_55_0 = Gemm_54_0.reshape((1, 17, 3, 16))
        Transpose_56_0 = Reshape_55_0.permute((0, 2, 1, 3))
        MatMul_57_0 = Transpose_56_0 @ Transpose_53_0
        Mul_58_0 = MatMul_57_0 * self.data['Initializer_17']
        Softmax_59_0 = self.Softmax_59(Mul_58_0)
        MatMul_60_0 = Softmax_59_0 @ Transpose_50_0
        Transpose_61_0 = MatMul_60_0.permute((0, 2, 1, 3))
        Reshape_62_0 = Transpose_61_0.reshape((1, 17, 48))
        Gemm_63_0 = self.Gemm_63(Reshape_62_0)
        Add_64_0 = Gemm_63_0 + Add_47_0
        Gemm_65_0 = self.Gemm_65(Add_64_0)
        Relu_66_0 = self.Relu_66(Gemm_65_0)
        Gemm_67_0 = self.Gemm_67(Relu_66_0)
        Add_68_0 = Gemm_67_0 + Add_64_0
        ReduceMean_69_0 = torch.mean(Add_68_0, dim=1)
        output_71 = self.Gemm_70(ReduceMean_69_0)

        return output_71

```

## 🤝 Contributing

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

📌 NOTE: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!
