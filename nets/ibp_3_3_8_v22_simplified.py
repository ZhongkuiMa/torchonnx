"""
This file was generated by the torchconverter.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


# Model name: torch_jit_slimmed
# Input name: input_0
# Input shape: [1, 3, 32, 32]
# Output name: output_71
# Output shape: [1, 10]


class Ibp338V22Simplified(Module):
    def __init__(self):
        super(Module, self).__init__()

        self.data = torch.load("../nets/ibp_3_3_8_v22_simplified.pth")

        self.Conv_1 = nn.Conv2d(3, 48, (8, 8), stride=8)
        self.Conv_1.weight.data = self.data["Initializer_1"]
        self.Conv_1.bias.data = self.data["Initializer_2"]

        self.Gemm_6 = nn.Linear(48, 48)
        self.Gemm_6.weight.data = self.data["Initializer_39"]
        self.Gemm_6.bias.data = self.data["Initializer_40"]

        self.Gemm_9 = nn.Linear(48, 48)
        self.Gemm_9.weight.data = self.data["Initializer_37"]
        self.Gemm_9.bias.data = self.data["Initializer_38"]

        self.Gemm_12 = nn.Linear(48, 48)
        self.Gemm_12.weight.data = self.data["Initializer_35"]
        self.Gemm_12.bias.data = self.data["Initializer_36"]

        self.Softmax_17 = nn.Softmax(dim=-1)
        self.Gemm_21 = nn.Linear(48, 48)
        self.Gemm_21.weight.data = self.data["Initializer_9"]
        self.Gemm_21.bias.data = self.data["Initializer_3"]

        self.Gemm_23 = nn.Linear(48, 96)
        self.Gemm_23.weight.data = self.data["Initializer_41"]
        self.Gemm_23.bias.data = self.data["Initializer_42"]

        self.Relu_24 = nn.ReLU()
        self.Gemm_25 = nn.Linear(96, 48)
        self.Gemm_25.weight.data = self.data["Initializer_10"]
        self.Gemm_25.bias.data = self.data["Initializer_4"]

        self.Gemm_27 = nn.Linear(48, 48)
        self.Gemm_27.weight.data = self.data["Initializer_47"]
        self.Gemm_27.bias.data = self.data["Initializer_48"]

        self.Gemm_30 = nn.Linear(48, 48)
        self.Gemm_30.weight.data = self.data["Initializer_45"]
        self.Gemm_30.bias.data = self.data["Initializer_46"]

        self.Gemm_33 = nn.Linear(48, 48)
        self.Gemm_33.weight.data = self.data["Initializer_43"]
        self.Gemm_33.bias.data = self.data["Initializer_44"]

        self.Softmax_38 = nn.Softmax(dim=-1)
        self.Gemm_42 = nn.Linear(48, 48)
        self.Gemm_42.weight.data = self.data["Initializer_11"]
        self.Gemm_42.bias.data = self.data["Initializer_5"]

        self.Gemm_44 = nn.Linear(48, 96)
        self.Gemm_44.weight.data = self.data["Initializer_49"]
        self.Gemm_44.bias.data = self.data["Initializer_50"]

        self.Relu_45 = nn.ReLU()
        self.Gemm_46 = nn.Linear(96, 48)
        self.Gemm_46.weight.data = self.data["Initializer_12"]
        self.Gemm_46.bias.data = self.data["Initializer_6"]

        self.Gemm_48 = nn.Linear(48, 48)
        self.Gemm_48.weight.data = self.data["Initializer_55"]
        self.Gemm_48.bias.data = self.data["Initializer_56"]

        self.Gemm_51 = nn.Linear(48, 48)
        self.Gemm_51.weight.data = self.data["Initializer_53"]
        self.Gemm_51.bias.data = self.data["Initializer_54"]

        self.Gemm_54 = nn.Linear(48, 48)
        self.Gemm_54.weight.data = self.data["Initializer_51"]
        self.Gemm_54.bias.data = self.data["Initializer_52"]

        self.Softmax_59 = nn.Softmax(dim=-1)
        self.Gemm_63 = nn.Linear(48, 48)
        self.Gemm_63.weight.data = self.data["Initializer_13"]
        self.Gemm_63.bias.data = self.data["Initializer_7"]

        self.Gemm_65 = nn.Linear(48, 96)
        self.Gemm_65.weight.data = self.data["Initializer_57"]
        self.Gemm_65.bias.data = self.data["Initializer_58"]

        self.Relu_66 = nn.ReLU()
        self.Gemm_67 = nn.Linear(96, 48)
        self.Gemm_67.weight.data = self.data["Initializer_14"]
        self.Gemm_67.bias.data = self.data["Initializer_8"]

        self.Gemm_70 = nn.Linear(48, 10)
        self.Gemm_70.weight.data = self.data["Initializer_33"]
        self.Gemm_70.bias.data = self.data["Initializer_34"]

    def forward(self, input_0: Tensor) -> Tensor:

        Conv_1 = self.Conv_1(input_0)
        Reshape_2 = Conv_1.reshape((1, 48, -1))
        Transpose_3 = Reshape_2.permute((0, 2, 1))
        Concat_4 = torch.cat([self.data["Initializer_20"], Transpose_3], dim=1)
        Add_5 = Concat_4 + self.data["Initializer_0"]
        Gemm_6 = self.Gemm_6(Add_5)
        Reshape_7 = Gemm_6.reshape((1, -1, 3, 16))
        Transpose_8 = Reshape_7.permute((0, 2, 1, 3))
        Gemm_9 = self.Gemm_9(Add_5)
        Reshape_10 = Gemm_9.reshape((1, -1, 3, 16))
        Transpose_11 = Reshape_10.permute((0, 2, 3, 1))
        Gemm_12 = self.Gemm_12(Add_5)
        Reshape_13 = Gemm_12.reshape((1, -1, 3, 16))
        Transpose_14 = Reshape_13.permute((0, 2, 1, 3))
        MatMul_15 = torch.matmul(Transpose_14, Transpose_11)
        Mul_16 = MatMul_15 * self.data["Initializer_15"]
        Softmax_17 = self.Softmax_17(Mul_16)
        MatMul_18 = torch.matmul(Softmax_17, Transpose_8)
        Transpose_19 = MatMul_18.permute((0, 2, 1, 3))
        Reshape_20 = Transpose_19.reshape((1, -1, 48))
        Gemm_21 = self.Gemm_21(Reshape_20)
        Add_22 = Gemm_21 + Add_5
        Gemm_23 = self.Gemm_23(Add_22)
        Relu_24 = self.Relu_24(Gemm_23)
        Gemm_25 = self.Gemm_25(Relu_24)
        Add_26 = Gemm_25 + Add_22
        Gemm_27 = self.Gemm_27(Add_26)
        Reshape_28 = Gemm_27.reshape((1, -1, 3, 16))
        Transpose_29 = Reshape_28.permute((0, 2, 1, 3))
        Gemm_30 = self.Gemm_30(Add_26)
        Reshape_31 = Gemm_30.reshape((1, -1, 3, 16))
        Transpose_32 = Reshape_31.permute((0, 2, 3, 1))
        Gemm_33 = self.Gemm_33(Add_26)
        Reshape_34 = Gemm_33.reshape((1, -1, 3, 16))
        Transpose_35 = Reshape_34.permute((0, 2, 1, 3))
        MatMul_36 = torch.matmul(Transpose_35, Transpose_32)
        Mul_37 = MatMul_36 * self.data["Initializer_16"]
        Softmax_38 = self.Softmax_38(Mul_37)
        MatMul_39 = torch.matmul(Softmax_38, Transpose_29)
        Transpose_40 = MatMul_39.permute((0, 2, 1, 3))
        Reshape_41 = Transpose_40.reshape((1, -1, 48))
        Gemm_42 = self.Gemm_42(Reshape_41)
        Add_43 = Gemm_42 + Add_26
        Gemm_44 = self.Gemm_44(Add_43)
        Relu_45 = self.Relu_45(Gemm_44)
        Gemm_46 = self.Gemm_46(Relu_45)
        Add_47 = Gemm_46 + Add_43
        Gemm_48 = self.Gemm_48(Add_47)
        Reshape_49 = Gemm_48.reshape((1, -1, 3, 16))
        Transpose_50 = Reshape_49.permute((0, 2, 1, 3))
        Gemm_51 = self.Gemm_51(Add_47)
        Reshape_52 = Gemm_51.reshape((1, -1, 3, 16))
        Transpose_53 = Reshape_52.permute((0, 2, 3, 1))
        Gemm_54 = self.Gemm_54(Add_47)
        Reshape_55 = Gemm_54.reshape((1, -1, 3, 16))
        Transpose_56 = Reshape_55.permute((0, 2, 1, 3))
        MatMul_57 = torch.matmul(Transpose_56, Transpose_53)
        Mul_58 = MatMul_57 * self.data["Initializer_17"]
        Softmax_59 = self.Softmax_59(Mul_58)
        MatMul_60 = torch.matmul(Softmax_59, Transpose_50)
        Transpose_61 = MatMul_60.permute((0, 2, 1, 3))
        Reshape_62 = Transpose_61.reshape((1, -1, 48))
        Gemm_63 = self.Gemm_63(Reshape_62)
        Add_64 = Gemm_63 + Add_47
        Gemm_65 = self.Gemm_65(Add_64)
        Relu_66 = self.Relu_66(Gemm_65)
        Gemm_67 = self.Gemm_67(Relu_66)
        Add_68 = Gemm_67 + Add_64
        ReduceMean_69 = torch.mean(Add_68, dim=1)
        output_71 = self.Gemm_70(ReduceMean_69)

        return output_71
