"""Synthetic ONNX model builders for comprehensive test coverage.

This module provides factory methods for creating various ONNX models
for testing different components of the torchonnx pipeline.
"""

import numpy as np
import onnx
import onnx.helper as onnx_helper


class SyntheticONNXModels:
    """Factory for creating synthetic ONNX models for testing."""

    # ===== Basic Models (Existing) =====

    @staticmethod
    def create_identity_model(input_shape=(1, 3), output_shape=(1, 3)):
        """Create minimal Identity ONNX model (pass-through).

        :param input_shape: Input tensor shape
        :param output_shape: Output tensor shape
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, input_shape
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, output_shape
        )

        node = onnx_helper.make_node("Identity", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "IdentityModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_add_model(shape=(2, 3)):
        """Create Add ONNX model with two inputs.

        :param shape: Tensor shape
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Z = onnx_helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape)  # noqa: N806

        node = onnx_helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])

        graph = onnx_helper.make_graph(
            [node],
            "AddModel",
            [X, Y],
            [Z],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_linear_model(input_size=3, output_size=2):
        """Create Linear (Gemm) ONNX model.

        :param input_size: Input size
        :param output_size: Output size
        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, input_size]
        )
        W_init = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [output_size, input_size],
            vals=rng.standard_normal((output_size, input_size))
            .astype(np.float32)
            .flatten()
            .tolist(),
        )
        B_init = onnx_helper.make_tensor(  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [output_size],
            vals=rng.standard_normal(output_size).astype(np.float32).tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, output_size]
        )

        node = onnx_helper.make_node(
            "Gemm",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            alpha=1.0,
            beta=1.0,
            transB=1,
        )

        graph = onnx_helper.make_graph(
            [node],
            "LinearModel",
            [X],
            [Y],
            [W_init, B_init],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_mlp_model():
        """Create simple 2-layer MLP model.

        Structure: Linear(3->4) -> ReLU -> Linear(4->2)

        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])  # noqa: N806

        W1 = onnx_helper.make_tensor(  # noqa: N806
            "W1",
            onnx.TensorProto.FLOAT,
            [4, 3],
            vals=rng.standard_normal((4, 3)).astype(np.float32).flatten().tolist(),
        )
        B1 = onnx_helper.make_tensor(  # noqa: N806
            "B1",
            onnx.TensorProto.FLOAT,
            [4],
            vals=rng.standard_normal(4).astype(np.float32).tolist(),
        )
        W2 = onnx_helper.make_tensor(  # noqa: N806
            "W2",
            onnx.TensorProto.FLOAT,
            [2, 4],
            vals=rng.standard_normal((2, 4)).astype(np.float32).flatten().tolist(),
        )
        B2 = onnx_helper.make_tensor(  # noqa: N806
            "B2",
            onnx.TensorProto.FLOAT,
            [2],
            vals=rng.standard_normal(2).astype(np.float32).tolist(),
        )

        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2])  # noqa: N806

        node1 = onnx_helper.make_node(
            "Gemm",
            inputs=["X", "W1", "B1"],
            outputs=["H"],
            alpha=1.0,
            beta=1.0,
            transB=1,
        )
        node2 = onnx_helper.make_node("Relu", inputs=["H"], outputs=["H_relu"])
        node3 = onnx_helper.make_node(
            "Gemm",
            inputs=["H_relu", "W2", "B2"],
            outputs=["Y"],
            alpha=1.0,
            beta=1.0,
            transB=1,
        )

        graph = onnx_helper.make_graph(
            [node1, node2, node3],
            "MLPModel",
            [X],
            [Y],
            [W1, B1, W2, B2],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Convolutional Models (New) =====

    @staticmethod
    def create_conv2d_model(in_channels=3, out_channels=16, kernel_size=3, stride=1, pads=1):
        """Create Conv2d ONNX model with padding, stride, dilation.

        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size
        :param stride: Stride
        :param pads: Padding (symmetric)
        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        # Input: [batch, channels, height, width] = [1, 3, 224, 224]
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, in_channels, 224, 224]
        )

        # Weight: [out_channels, in_channels, kernel_h, kernel_w]
        W = onnx_helper.make_tensor(  # noqa: N806  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [out_channels, in_channels, kernel_size, kernel_size],
            vals=rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size))
            .astype(np.float32)
            .flatten()
            .tolist(),
        )

        # Bias: [out_channels]
        B = onnx_helper.make_tensor(  # noqa: N806  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [out_channels],
            vals=rng.standard_normal(out_channels).astype(np.float32).tolist(),
        )

        # Output
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, out_channels, 224, 224]
        )

        # Conv node
        node = onnx_helper.make_node(
            "Conv",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=[kernel_size, kernel_size],
            pads=[pads, pads, pads, pads],
            strides=[stride, stride],
        )

        graph = onnx_helper.make_graph(
            [node],
            "Conv2dModel",
            [X],
            [Y],
            [W, B],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_batchnorm_model(num_channels=16, spatial_dims=2):
        """Create BatchNorm ONNX model with running statistics.

        :param num_channels: Number of channels
        :param spatial_dims: Number of spatial dimensions (2 for 2D, 3 for 3D)
        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)

        if spatial_dims == 2:
            input_shape = [1, num_channels, 224, 224]
        else:
            input_shape = [1, num_channels, 32, 32, 32]

        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, input_shape
        )

        # BatchNorm parameters
        scale = onnx_helper.make_tensor(
            "scale",
            onnx.TensorProto.FLOAT,
            [num_channels],
            vals=np.ones(num_channels, dtype=np.float32).tolist(),
        )
        bias = onnx_helper.make_tensor(
            "bias",
            onnx.TensorProto.FLOAT,
            [num_channels],
            vals=np.zeros(num_channels, dtype=np.float32).tolist(),
        )
        mean = onnx_helper.make_tensor(
            "mean",
            onnx.TensorProto.FLOAT,
            [num_channels],
            vals=rng.standard_normal(num_channels).astype(np.float32).tolist(),
        )
        var = onnx_helper.make_tensor(
            "var",
            onnx.TensorProto.FLOAT,
            [num_channels],
            vals=np.ones(num_channels, dtype=np.float32).tolist(),
        )

        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, input_shape
        )

        node = onnx_helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y"],
            epsilon=1e-5,
            momentum=0.1,
        )

        graph = onnx_helper.make_graph(
            [node],
            "BatchNormModel",
            [X],
            [Y],
            [scale, bias, mean, var],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Pooling Models (New) =====

    @staticmethod
    def create_maxpool_model(kernel_size=2, stride=2):
        """Create MaxPool2d ONNX model.

        :param kernel_size: Kernel size
        :param stride: Stride
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 16, 224, 224]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 16, 112, 112]
        )

        node = onnx_helper.make_node(
            "MaxPool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[kernel_size, kernel_size],
            strides=[stride, stride],
        )

        graph = onnx_helper.make_graph(
            [node],
            "MaxPoolModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_avgpool_model(kernel_size=7):
        """Create AvgPool ONNX model.

        :param kernel_size: Kernel size
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 512, 7, 7]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 512, 1, 1]
        )

        node = onnx_helper.make_node(
            "AveragePool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[kernel_size, kernel_size],
        )

        graph = onnx_helper.make_graph(
            [node],
            "AvgPoolModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Shape Operation Models (New) =====

    @staticmethod
    def create_reshape_model():
        """Create Reshape ONNX model.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 512]
        )

        # Shape constant: [1, 512, 1, 1]
        shape_const = onnx_helper.make_tensor(
            "shape",
            onnx.TensorProto.INT64,
            [4],
            vals=np.array([1, 512, 1, 1], dtype=np.int64).tolist(),
        )

        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 512, 1, 1]
        )

        node = onnx_helper.make_node(
            "Reshape",
            inputs=["X", "shape"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "ReshapeModel",
            [X],
            [Y],
            [shape_const],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_concat_model():
        """Create Concat ONNX model.

        :return: ONNX ModelProto
        """
        X1 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X1", onnx.TensorProto.FLOAT, [1, 256, 56, 56]
        )
        X2 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X2", onnx.TensorProto.FLOAT, [1, 256, 56, 56]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 512, 56, 56]
        )

        node = onnx_helper.make_node(
            "Concat",
            inputs=["X1", "X2"],
            outputs=["Y"],
            axis=1,
        )

        graph = onnx_helper.make_graph(
            [node],
            "ConcatModel",
            [X1, X2],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_transpose_model():
        """Create Transpose ONNX model.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 224, 224, 3]
        )

        node = onnx_helper.make_node(
            "Transpose",
            inputs=["X"],
            outputs=["Y"],
            perm=[0, 2, 3, 1],
        )

        graph = onnx_helper.make_graph(
            [node],
            "TransposeModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Reduction Models (New) =====

    @staticmethod
    def create_reduce_mean_model():
        """Create ReduceMean ONNX model.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 512, 7, 7]
        )
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [2], vals=[2, 3])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 512, 1, 1]
        )

        node = onnx_helper.make_node(
            "ReduceMean",
            inputs=["X", "axes"],
            outputs=["Y"],
            keepdims=1,
        )

        graph = onnx_helper.make_graph(
            [node],
            "ReduceMeanModel",
            [X],
            [Y],
            [axes],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_reduce_sum_model():
        """Create ReduceSum ONNX model.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10, 10]
        )
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 1, 10]
        )

        node = onnx_helper.make_node(
            "ReduceSum",
            inputs=["X", "axes"],
            outputs=["Y"],
            keepdims=1,
        )

        graph = onnx_helper.make_graph(
            [node],
            "ReduceSumModel",
            [X],
            [Y],
            [axes],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Operator Models (New) =====

    @staticmethod
    def create_arithmetic_model():
        """Create Arithmetic operations model (Add, Sub, Mul, Div).

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )
        Z = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Z", onnx.TensorProto.FLOAT, [1, 10]
        )

        node_add = onnx_helper.make_node("Add", inputs=["X", "Y"], outputs=["sum"])
        node_mul = onnx_helper.make_node("Mul", inputs=["sum", "Y"], outputs=["Z"])

        graph = onnx_helper.make_graph(
            [node_add, node_mul],
            "ArithmeticModel",
            [X, Y],
            [Z],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_matmul_model():
        """Create MatMul ONNX model.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10, 20]
        )
        rng = np.random.default_rng()
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [20, 30],
            vals=rng.standard_normal((20, 30)).astype(np.float32).flatten().tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10, 30]
        )

        node = onnx_helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "MatMulModel",
            [X],
            [Y],
            [W],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_sub_model(shape=(2, 3)):
        """Create Sub ONNX model with two inputs.

        :param shape: Tensor shape
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Z = onnx_helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape)  # noqa: N806

        node = onnx_helper.make_node("Sub", inputs=["X", "Y"], outputs=["Z"])

        graph = onnx_helper.make_graph(
            [node],
            "SubModel",
            [X, Y],
            [Z],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_div_model(shape=(2, 3)):
        """Create Div ONNX model with two inputs.

        :param shape: Tensor shape
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Z = onnx_helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape)  # noqa: N806

        node = onnx_helper.make_node("Div", inputs=["X", "Y"], outputs=["Z"])

        graph = onnx_helper.make_graph(
            [node],
            "DivModel",
            [X, Y],
            [Z],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_pow_model(shape=(2, 3)):
        """Create Pow ONNX model with two inputs.

        :param shape: Tensor shape
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Z = onnx_helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape)  # noqa: N806

        node = onnx_helper.make_node("Pow", inputs=["X", "Y"], outputs=["Z"])

        graph = onnx_helper.make_graph(
            [node],
            "PowModel",
            [X, Y],
            [Z],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_neg_model(shape=(2, 3)):
        """Create Neg ONNX model (unary negation).

        :param shape: Tensor shape
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)  # noqa: N806

        node = onnx_helper.make_node("Neg", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "NegModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_equal_model(shape=(2, 3)):
        """Create Equal ONNX model with two inputs.

        :param shape: Tensor shape
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)  # noqa: N806
        Z = onnx_helper.make_tensor_value_info("Z", onnx.TensorProto.BOOL, shape)  # noqa: N806

        node = onnx_helper.make_node("Equal", inputs=["X", "Y"], outputs=["Z"])

        graph = onnx_helper.make_graph(
            [node],
            "EqualModel",
            [X, Y],
            [Z],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_add_with_vector_constant():
        """Create Add ONNX model with small vector constant.

        Tests vector literal path in _get_input_code_name.
        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 5]
        )
        # Small vector constant (5 elements, triggering vector literal path if enabled)
        bias = onnx_helper.make_tensor(
            "bias",
            onnx.TensorProto.FLOAT,
            [5],
            vals=rng.standard_normal(5).astype(np.float32).tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 5]
        )

        node = onnx_helper.make_node("Add", inputs=["X", "bias"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "AddVectorConstantModel",
            [X],
            [Y],
            [bias],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_mul_with_scalar_constant():
        """Create Mul ONNX model with scalar constant.

        Tests scalar literal path in _get_input_code_name.
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 5]
        )
        # Scalar constant
        scale = onnx_helper.make_tensor(
            "scale",
            onnx.TensorProto.FLOAT,
            [1],
            vals=[2.5],
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 5]
        )

        node = onnx_helper.make_node("Mul", inputs=["X", "scale"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "MulScalarConstantModel",
            [X],
            [Y],
            [scale],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_chained_operators_model():
        """Create model with chained operators and parameters.

        Tests parameter marking in multiple operator contexts.
        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 5]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 5]
        )

        # Create a chain: X -> Sub -> Mul -> Add -> Y
        weight1 = onnx_helper.make_tensor(
            "weight1",
            onnx.TensorProto.FLOAT,
            [5],
            vals=rng.standard_normal(5).astype(np.float32).tolist(),
        )
        weight2 = onnx_helper.make_tensor(
            "weight2",
            onnx.TensorProto.FLOAT,
            [1],
            vals=[1.5],
        )
        weight3 = onnx_helper.make_tensor(
            "weight3",
            onnx.TensorProto.FLOAT,
            [5],
            vals=rng.standard_normal(5).astype(np.float32).tolist(),
        )

        sub_node = onnx_helper.make_node("Sub", inputs=["X", "weight1"], outputs=["sub_out"])
        mul_node = onnx_helper.make_node("Mul", inputs=["sub_out", "weight2"], outputs=["mul_out"])
        add_node = onnx_helper.make_node("Add", inputs=["mul_out", "weight3"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [sub_node, mul_node, add_node],
            "ChainedOperatorsModel",
            [X],
            [Y],
            [weight1, weight2, weight3],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_reshape_with_shape_tensor():
        """Create Reshape model with shape tensor input.

        Tests reshape with runtime shape tensor.
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 20]
        )
        # Shape tensor as input
        shape_tensor = onnx_helper.make_tensor(
            "shape",
            onnx.TensorProto.INT64,
            [2],
            vals=[4, 5],
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [4, 5]
        )

        node = onnx_helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "ReshapeShapeTensorModel",
            [X],
            [Y],
            [shape_tensor],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_gather_with_axis():
        """Create Gather model with different axis.

        Tests gather along specific axis.
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [2, 3, 4]
        )
        indices = onnx_helper.make_tensor(
            "indices",
            onnx.TensorProto.INT64,
            [2],
            vals=[0, 2],
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [2, 2, 4]
        )

        node = onnx_helper.make_node("Gather", inputs=["X", "indices"], outputs=["Y"], axis=1)

        graph = onnx_helper.make_graph(
            [node],
            "GatherAxisModel",
            [X],
            [Y],
            [indices],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_multi_concat_model():
        """Create Concat model with 3 inputs.

        Tests concatenation of multiple tensors.
        :return: ONNX ModelProto
        """
        X1 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X1", onnx.TensorProto.FLOAT, [2, 3]
        )
        X2 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X2", onnx.TensorProto.FLOAT, [2, 3]
        )
        X3 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X3", onnx.TensorProto.FLOAT, [2, 3]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [2, 9]
        )

        node = onnx_helper.make_node("Concat", inputs=["X1", "X2", "X3"], outputs=["Y"], axis=1)

        graph = onnx_helper.make_graph(
            [node],
            "MultiConcatModel",
            [X1, X2, X3],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_reduce_with_keepdims():
        """Create ReduceMean model with keepdims.

        Tests reduce operation with keepdims=1.
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [2, 3, 4, 5]
        )
        axes = onnx_helper.make_tensor(
            "axes",
            onnx.TensorProto.INT64,
            [1],
            vals=[1],
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [2, 1, 4, 5]
        )

        node = onnx_helper.make_node("ReduceMean", inputs=["X", "axes"], outputs=["Y"], keepdims=1)

        graph = onnx_helper.make_graph(
            [node],
            "ReduceKeepdims",
            [X],
            [Y],
            [axes],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_expand_with_runtime_shape():
        """Create Expand model with runtime shape.

        Tests expand with dynamic shape.
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 3, 1]
        )
        shape = onnx_helper.make_tensor(
            "shape",
            onnx.TensorProto.INT64,
            [3],
            vals=[2, 3, 4],
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [2, 3, 4]
        )

        node = onnx_helper.make_node("Expand", inputs=["X", "shape"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "ExpandRuntimeShape",
            [X],
            [Y],
            [shape],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_split_equal_parts_model():
        """Create Split model with equal parts.

        Tests split operation with equal split sizes.
        Uses opset 13+ where split is an input tensor, not an attribute.
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 12]
        )
        Y1 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y1", onnx.TensorProto.FLOAT, [1, 4]
        )
        Y2 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y2", onnx.TensorProto.FLOAT, [1, 4]
        )
        Y3 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y3", onnx.TensorProto.FLOAT, [1, 4]
        )

        # Split sizes as input tensor (opset 13+)
        split_tensor = onnx_helper.make_tensor(
            "split_sizes",
            onnx.TensorProto.INT64,
            [3],
            vals=[4, 4, 4],
        )

        # Split equally: 4, 4, 4
        node = onnx_helper.make_node(
            "Split", inputs=["X", "split_sizes"], outputs=["Y1", "Y2", "Y3"], axis=1
        )

        graph = onnx_helper.make_graph(
            [node],
            "SplitEqualModel",
            [X],
            [Y1, Y2, Y3],
            [split_tensor],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model

    # ===== Complex Models (New) =====

    @staticmethod
    def create_multi_input_model():
        """Create model with multiple inputs.

        :return: ONNX ModelProto
        """
        X1 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X1", onnx.TensorProto.FLOAT, [1, 10]
        )
        X2 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X2", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node("Add", inputs=["X1", "X2"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "MultiInputModel",
            [X1, X2],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_multi_output_model():
        """Create model with multiple outputs.

        Uses Split with split attribute for equal split (opset 11).
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 20]
        )
        Y1 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y1", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y2 = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y2", onnx.TensorProto.FLOAT, [1, 10]
        )

        # Split node with split attribute for equal split (opset 11 compatible)
        node = onnx_helper.make_node(
            "Split", inputs=["X"], outputs=["Y1", "Y2"], axis=1, split=[10, 10]
        )

        graph = onnx_helper.make_graph(
            [node],
            "MultiOutputModel",
            [X],
            [Y1, Y2],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 11)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_resnet_block():
        """Create simplified ResNet block with skip connection.

        Structure: Conv -> BatchNorm -> ReLU + Skip -> Add

        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)

        # Input and output
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 64, 56, 56]
        )

        # Conv parameters
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [64, 64, 3, 3],
            vals=rng.standard_normal((64, 64, 3, 3)).astype(np.float32).flatten().tolist(),
        )
        B = onnx_helper.make_tensor(  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [64],
            vals=rng.standard_normal(64).astype(np.float32).tolist(),
        )

        # BatchNorm parameters
        scale = onnx_helper.make_tensor(
            "scale",
            onnx.TensorProto.FLOAT,
            [64],
            vals=np.ones(64, dtype=np.float32).tolist(),
        )
        bias = onnx_helper.make_tensor(
            "bias",
            onnx.TensorProto.FLOAT,
            [64],
            vals=np.zeros(64, dtype=np.float32).tolist(),
        )
        mean = onnx_helper.make_tensor(
            "mean",
            onnx.TensorProto.FLOAT,
            [64],
            vals=rng.standard_normal(64).astype(np.float32).tolist(),
        )
        var = onnx_helper.make_tensor(
            "var",
            onnx.TensorProto.FLOAT,
            [64],
            vals=np.ones(64, dtype=np.float32).tolist(),
        )

        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 64, 56, 56]
        )

        # Nodes
        node_conv = onnx_helper.make_node(
            "Conv",
            inputs=["X", "W", "B"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        node_bn = onnx_helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "scale", "bias", "mean", "var"],
            outputs=["bn_out"],
        )
        node_relu = onnx_helper.make_node("Relu", inputs=["bn_out"], outputs=["relu_out"])
        node_add = onnx_helper.make_node("Add", inputs=["relu_out", "X"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node_conv, node_bn, node_relu, node_add],
            "ResNetBlock",
            [X],
            [Y],
            [W, B, scale, bias, mean, var],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Error Test Models (New) =====

    @staticmethod
    def create_asymmetric_padding_model():
        """Create Conv2d with asymmetric padding (should fail conversion).

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
        )
        rng = np.random.default_rng()
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [16, 3, 3, 3],
            vals=rng.standard_normal((16, 3, 3, 3)).astype(np.float32).flatten().tolist(),
        )
        B = onnx_helper.make_tensor(  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [16],
            vals=rng.standard_normal(16).astype(np.float32).tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 16, 224, 224]
        )

        # Asymmetric padding: [1, 2, 1, 2] instead of [1, 1, 1, 1]
        node = onnx_helper.make_node(
            "Conv",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            pads=[1, 2, 1, 2],  # Asymmetric!
            strides=[1, 1],
        )

        graph = onnx_helper.make_graph(
            [node],
            "AsymmetricPaddingModel",
            [X],
            [Y],
            [W, B],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_unsupported_op_model():
        """Create model with unsupported operator.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        # Use an operator that may not be supported
        node = onnx_helper.make_node("QuantizeLinear", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "UnsupportedOpModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_constant_node_model():
        """Create model with Constant nodes (should be folded).

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )

        # Constant node
        const_tensor = onnx.numpy_helper.from_array(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
            name="const_val",
        )
        node_const = onnx_helper.make_node("Constant", inputs=[], outputs=["C"], value=const_tensor)

        # Add node
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )
        node_add = onnx_helper.make_node("Add", inputs=["X", "C"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node_const, node_add],
            "ConstantNodeModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Phase 3: Additional Operator Models =====

    @staticmethod
    def create_cast_model(to_dtype=onnx.TensorProto.FLOAT):
        """Create Cast ONNX model for dtype conversion.

        :param to_dtype: Target dtype for Cast operation
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node(
            "Cast",
            inputs=["X"],
            outputs=["Y"],
            to=to_dtype,
        )

        graph = onnx_helper.make_graph(
            [node],
            "CastModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_argmax_model(axis=1, keepdims=1):
        """Create ArgMax ONNX model for reduction.

        :param axis: Axis for ArgMax
        :param keepdims: Whether to keep dimension
        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.INT64, [1, 1] if keepdims else [1]
        )

        node = onnx_helper.make_node(
            "ArgMax",
            inputs=["X"],
            outputs=["Y"],
            axis=axis,
            keepdims=keepdims,
        )

        graph = onnx_helper.make_graph(
            [node],
            "ArgMaxModel",
            [X],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_convtranspose_model(in_channels=3, out_channels=16, kernel_size=3, stride=2, pad=1):
        """Create ConvTranspose ONNX model.

        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size
        :param stride: Stride
        :param pad: Padding
        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, in_channels, 8, 8]
        )
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [in_channels, out_channels, kernel_size, kernel_size],
            vals=rng.standard_normal((in_channels, out_channels, kernel_size, kernel_size))
            .astype(np.float32)
            .flatten()
            .tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, out_channels, 16, 16]
        )

        node = onnx_helper.make_node(
            "ConvTranspose",
            inputs=["X", "W"],
            outputs=["Y"],
            kernel_shape=[kernel_size, kernel_size],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
        )

        graph = onnx_helper.make_graph(
            [node],
            "ConvTransposeModel",
            [X],
            [Y],
            [W],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_constantofshape_model(shape=(1, 10), value=1.0):
        """Create ConstantOfShape ONNX model.

        :param shape: Output shape
        :param value: Constant value
        :return: ONNX ModelProto
        """
        # Create shape tensor
        shape_tensor = onnx_helper.make_tensor(
            "shape_tensor",
            onnx.TensorProto.INT64,
            [len(shape)],
            vals=list(shape),
        )

        # Create value tensor for ConstantOfShape attribute
        value_tensor = onnx_helper.make_tensor(
            "value_tensor",
            onnx.TensorProto.FLOAT,
            [1],
            vals=[value],
        )

        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, shape
        )

        node = onnx_helper.make_node(
            "ConstantOfShape",
            inputs=["shape_tensor"],
            outputs=["Y"],
            value=value_tensor,
        )

        graph = onnx_helper.make_graph(
            [node],
            "ConstantOfShapeModel",
            [],
            [Y],
            [shape_tensor],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Phase 1: Slice Operation Models =====

    @staticmethod
    def create_slice_static_model():
        """Create Slice ONNX model with all static parameters.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20, 10])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10, 10])  # noqa: N806

        starts = onnx_helper.make_tensor("starts", onnx.TensorProto.INT64, [1], vals=[0])
        ends = onnx_helper.make_tensor("ends", onnx.TensorProto.INT64, [1], vals=[10])
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])
        steps = onnx_helper.make_tensor("steps", onnx.TensorProto.INT64, [1], vals=[1])

        node = onnx_helper.make_node(
            "Slice",
            inputs=["X", "starts", "ends", "axes", "steps"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "SliceStaticModel",
            [X],
            [Y],
            [starts, ends, axes, steps],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_slice_dynamic_starts_model():
        """Create Slice ONNX model with dynamic starts parameter.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20, 10])  # noqa: N806
        starts_input = onnx_helper.make_tensor_value_info("starts", onnx.TensorProto.INT64, [1])
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10, 10])  # noqa: N806

        ends = onnx_helper.make_tensor("ends", onnx.TensorProto.INT64, [1], vals=[10])
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])
        steps = onnx_helper.make_tensor("steps", onnx.TensorProto.INT64, [1], vals=[1])

        node = onnx_helper.make_node(
            "Slice",
            inputs=["X", "starts", "ends", "axes", "steps"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "SliceDynamicStartsModel",
            [X, starts_input],
            [Y],
            [ends, axes, steps],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_slice_dynamic_ends_model():
        """Create Slice ONNX model with dynamic ends parameter.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20, 10])  # noqa: N806
        ends_input = onnx_helper.make_tensor_value_info("ends", onnx.TensorProto.INT64, [1])
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10, 10])  # noqa: N806

        starts = onnx_helper.make_tensor("starts", onnx.TensorProto.INT64, [1], vals=[0])
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])
        steps = onnx_helper.make_tensor("steps", onnx.TensorProto.INT64, [1], vals=[1])

        node = onnx_helper.make_node(
            "Slice",
            inputs=["X", "starts", "ends", "axes", "steps"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "SliceDynamicEndsModel",
            [X, ends_input],
            [Y],
            [starts, axes, steps],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_slice_narrow_compatible_model():
        """Create Slice ONNX model optimizable to narrow operation.

        Single axis slice that could use torch.narrow() optimization.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 5])  # noqa: N806

        starts = onnx_helper.make_tensor("starts", onnx.TensorProto.INT64, [1], vals=[5])
        ends = onnx_helper.make_tensor("ends", onnx.TensorProto.INT64, [1], vals=[10])
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])
        steps = onnx_helper.make_tensor("steps", onnx.TensorProto.INT64, [1], vals=[1])

        node = onnx_helper.make_node(
            "Slice",
            inputs=["X", "starts", "ends", "axes", "steps"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "SliceNarrowCompatibleModel",
            [X],
            [Y],
            [starts, ends, axes, steps],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_slice_multi_axis_model():
        """Create Slice ONNX model slicing multiple axes.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20, 15])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10, 5])  # noqa: N806

        starts = onnx_helper.make_tensor("starts", onnx.TensorProto.INT64, [2], vals=[0, 5])
        ends = onnx_helper.make_tensor("ends", onnx.TensorProto.INT64, [2], vals=[10, 10])
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [2], vals=[1, 2])
        steps = onnx_helper.make_tensor("steps", onnx.TensorProto.INT64, [2], vals=[1, 1])

        node = onnx_helper.make_node(
            "Slice",
            inputs=["X", "starts", "ends", "axes", "steps"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "SliceMultiAxisModel",
            [X],
            [Y],
            [starts, ends, axes, steps],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_slice_int64max_model():
        """Create Slice ONNX model with INT64_MAX end value (should omit end).

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20, 10])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 15, 10])  # noqa: N806

        starts = onnx_helper.make_tensor("starts", onnx.TensorProto.INT64, [1], vals=[5])
        ends = onnx_helper.make_tensor(
            "ends", onnx.TensorProto.INT64, [1], vals=[np.iinfo(np.int64).max]
        )
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])
        steps = onnx_helper.make_tensor("steps", onnx.TensorProto.INT64, [1], vals=[1])

        node = onnx_helper.make_node(
            "Slice",
            inputs=["X", "starts", "ends", "axes", "steps"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "SliceInt64MaxModel",
            [X],
            [Y],
            [starts, ends, axes, steps],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Phase 2: Expand & Pad Operation Models =====

    @staticmethod
    def create_expand_constant_shape_model():
        """Create Expand ONNX model with constant output shape.

        Expand broadcasts input to a larger shape.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 10])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 3, 10])  # noqa: N806

        shape = onnx_helper.make_tensor("shape", onnx.TensorProto.INT64, [3], vals=[2, 3, 10])

        node = onnx_helper.make_node(
            "Expand",
            inputs=["X", "shape"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "ExpandConstantShapeModel",
            [X],
            [Y],
            [shape],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_expand_runtime_shape_model():
        """Create Expand ONNX model with dynamic output shape.

        Shape is provided as input (runtime).

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 10])  # noqa: N806
        shape_input = onnx_helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [3])
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None, None, 10])  # noqa: N806

        node = onnx_helper.make_node(
            "Expand",
            inputs=["X", "shape"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "ExpandRuntimeShapeModel",
            [X, shape_input],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_expand_broadcast_model():
        """Create Expand ONNX model testing broadcasting semantics.

        Expand from [1, 10] to [5, 10] using ONNX broadcast semantics.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [5, 10])  # noqa: N806

        shape = onnx_helper.make_tensor("shape", onnx.TensorProto.INT64, [2], vals=[5, 10])

        node = onnx_helper.make_node(
            "Expand",
            inputs=["X", "shape"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "ExpandBroadcastModel",
            [X],
            [Y],
            [shape],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_pad_constant_pads_model():
        """Create Pad ONNX model with constant padding values.

        Pad with static pad amounts on each side.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 10, 10])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3, 12, 12])  # noqa: N806

        pads = onnx_helper.make_tensor(
            "pads", onnx.TensorProto.INT64, [8], vals=[0, 0, 1, 1, 0, 0, 1, 1]
        )

        node = onnx_helper.make_node(
            "Pad",
            inputs=["X", "pads"],
            outputs=["Y"],
            mode="constant",
        )

        graph = onnx_helper.make_graph(
            [node],
            "PadConstantPadsModel",
            [X],
            [Y],
            [pads],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_pad_dynamic_pads_model():
        """Create Pad ONNX model with dynamic padding values.

        Padding amounts provided as input (runtime).

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 10, 10])  # noqa: N806
        pads_input = onnx_helper.make_tensor_value_info("pads", onnx.TensorProto.INT64, [8])
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3, None, None])  # noqa: N806

        node = onnx_helper.make_node(
            "Pad",
            inputs=["X", "pads"],
            outputs=["Y"],
            mode="constant",
        )

        graph = onnx_helper.make_graph(
            [node],
            "PadDynamicPadsModel",
            [X, pads_input],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_pad_with_value_model():
        """Create Pad ONNX model with non-zero pad value.

        Padding with constant value other than 0.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 10, 10])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3, 12, 12])  # noqa: N806

        pads = onnx_helper.make_tensor(
            "pads", onnx.TensorProto.INT64, [8], vals=[0, 0, 1, 1, 0, 0, 1, 1]
        )
        value = onnx_helper.make_tensor("value", onnx.TensorProto.FLOAT, [1], vals=[0.5])

        node = onnx_helper.make_node(
            "Pad",
            inputs=["X", "pads", "value"],
            outputs=["Y"],
            mode="constant",
        )

        graph = onnx_helper.make_graph(
            [node],
            "PadWithValueModel",
            [X],
            [Y],
            [pads, value],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Phase 3: Indexing Operation Models =====

    @staticmethod
    def create_gather_scalar_index_model():
        """Create Gather ONNX model with scalar index.

        Gather with single index value.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [10, 5])  # noqa: N806
        indices = onnx_helper.make_tensor("indices", onnx.TensorProto.INT64, [1], vals=[2])
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 5])  # noqa: N806

        node = onnx_helper.make_node(
            "Gather",
            inputs=["X", "indices"],
            outputs=["Y"],
            axis=0,
        )

        graph = onnx_helper.make_graph(
            [node],
            "GatherScalarIndexModel",
            [X],
            [Y],
            [indices],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_gather_vector_indices_model():
        """Create Gather ONNX model with vector indices.

        Gather with multiple index values.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [10, 5])  # noqa: N806
        indices = onnx_helper.make_tensor("indices", onnx.TensorProto.INT64, [3], vals=[1, 3, 5])
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 5])  # noqa: N806

        node = onnx_helper.make_node(
            "Gather",
            inputs=["X", "indices"],
            outputs=["Y"],
            axis=0,
        )

        graph = onnx_helper.make_graph(
            [node],
            "GatherVectorIndicesModel",
            [X],
            [Y],
            [indices],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_scatter_nd_model():
        """Create ScatterND ONNX model.

        ScatterND updates elements using indices and updates.

        :return: ONNX ModelProto
        """
        data = onnx_helper.make_tensor(
            "data", onnx.TensorProto.FLOAT, [4, 4], vals=np.zeros(16, dtype=np.float32).tolist()
        )
        indices = onnx_helper.make_tensor(
            "indices", onnx.TensorProto.INT64, [2, 2], vals=[0, 0, 1, 1]
        )
        updates = onnx_helper.make_tensor("updates", onnx.TensorProto.FLOAT, [2], vals=[1.0, 2.0])
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 4])  # noqa: N806

        node = onnx_helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["Y"],
        )

        graph = onnx_helper.make_graph(
            [node],
            "ScatterNDModel",
            [],
            [Y],
            [data, indices, updates],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_concat_batch_expand_model():
        """Create Concat ONNX model for batch expansion.

        Concatenate constant tensors along batch dimension.

        :return: ONNX ModelProto
        """
        X1 = onnx_helper.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        X2 = onnx_helper.make_tensor_value_info("X2", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 10])  # noqa: N806

        node = onnx_helper.make_node(
            "Concat",
            inputs=["X1", "X2"],
            outputs=["Y"],
            axis=0,
        )

        graph = onnx_helper.make_graph(
            [node],
            "ConcatBatchExpandModel",
            [X1, X2],
            [Y],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_split_equal_model():
        """Create Split ONNX model with equal split sizes.

        Split tensor into equal chunks.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20])  # noqa: N806
        Y1 = onnx_helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y2 = onnx_helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

        node = onnx_helper.make_node(
            "Split",
            inputs=["X"],
            outputs=["Y1", "Y2"],
            axis=1,
            num_outputs=2,
        )

        graph = onnx_helper.make_graph(
            [node],
            "SplitEqualModel",
            [X],
            [Y1, Y2],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_split_unequal_model():
        """Create Split ONNX model with unequal split sizes.

        Split tensor into chunks of different sizes.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20])  # noqa: N806
        split = onnx_helper.make_tensor("split", onnx.TensorProto.INT64, [2], vals=[5, 15])
        Y1 = onnx_helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [1, 5])  # noqa: N806
        Y2 = onnx_helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [1, 15])  # noqa: N806

        node = onnx_helper.make_node(
            "Split",
            inputs=["X", "split"],
            outputs=["Y1", "Y2"],
            axis=1,
        )

        graph = onnx_helper.make_graph(
            [node],
            "SplitUnequalModel",
            [X],
            [Y1, Y2],
            [split],
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Phase 4: Convolution and Linear Operations =====

    @staticmethod
    def create_conv1d_model():
        """Create 1D Convolution ONNX model.

        Test conv dimension detection (3D input -> F.conv1d).

        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X",
            onnx.TensorProto.FLOAT,
            [1, 3, 16],  # (batch, channels, length)
        )
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [4, 3, 3],  # (out_channels, in_channels, kernel_size)
            vals=rng.standard_normal((4, 3, 3)).astype(np.float32).flatten().tolist(),
        )
        B = onnx_helper.make_tensor(  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [4],  # (out_channels,)
            vals=rng.standard_normal(4).astype(np.float32).tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y",
            onnx.TensorProto.FLOAT,
            [1, 4, 14],  # Output shape after conv
        )

        node = onnx_helper.make_node(
            "Conv",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=[3],
            pads=[0, 0],
            strides=[1],
        )

        graph = onnx_helper.make_graph([node], "Conv1DModel", [X], [Y], [W, B])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_conv3d_model():
        """Create 3D Convolution ONNX model.

        Test conv dimension detection (5D input -> F.conv3d).

        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X",
            onnx.TensorProto.FLOAT,
            [1, 3, 8, 8, 8],  # (batch, channels, d, h, w)
        )
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [4, 3, 3, 3, 3],  # (out_channels, in_channels, d, h, w)
            vals=rng.standard_normal((4, 3, 3, 3, 3)).astype(np.float32).flatten().tolist(),
        )
        B = onnx_helper.make_tensor(  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [4],  # (out_channels,)
            vals=rng.standard_normal(4).astype(np.float32).tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 4, 6, 6, 6]
        )

        node = onnx_helper.make_node(
            "Conv",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=[3, 3, 3],
            pads=[0, 0, 0, 0, 0, 0],
            strides=[1, 1, 1],
        )

        graph = onnx_helper.make_graph([node], "Conv3DModel", [X], [Y], [W, B])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_linear_transposed_model():
        """Create Linear (Gemm) ONNX model with transposed weights (transB=1).

        Test weight transposition handling in linear operations.

        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 3]
        )
        # Weight is transposed: shape [input_size, output_size] instead of [output_size, input_size]
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [3, 2],  # [input_size, output_size] (transposed)
            vals=rng.standard_normal((3, 2)).astype(np.float32).flatten().tolist(),
        )
        B = onnx_helper.make_tensor(  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [2],
            vals=rng.standard_normal(2).astype(np.float32).tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 2]
        )

        node = onnx_helper.make_node(
            "Gemm",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            transA=0,
            transB=1,  # Weight is transposed
            alpha=1.0,
            beta=1.0,
        )

        graph = onnx_helper.make_graph([node], "LinearTransposedModel", [X], [Y], [W, B])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_interpolate_model():
        """Create Interpolate (Resize) ONNX model.

        Test mode mapping and scale/size handling.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 3, 4, 4]
        )
        scales = onnx_helper.make_tensor(
            "scales", onnx.TensorProto.FLOAT, [4], vals=[1.0, 1.0, 2.0, 2.0]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        )

        node = onnx_helper.make_node(
            "Resize",
            inputs=["X", "", "scales"],  # roi is empty, scales provided
            outputs=["Y"],
            mode="linear",  # bilinear interpolation
        )

        graph = onnx_helper.make_graph([node], "InterpolateModel", [X], [Y], [scales])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_conv_transpose_model():
        """Create Transpose Convolution (ConvTranspose) ONNX model.

        Test transposed convolution operation.

        :return: ONNX ModelProto
        """
        rng = np.random.default_rng(42)
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X",
            onnx.TensorProto.FLOAT,
            [1, 3, 4, 4],  # (batch, channels, h, w)
        )
        W = onnx_helper.make_tensor(  # noqa: N806
            "W",
            onnx.TensorProto.FLOAT,
            [3, 4, 3, 3],  # (in_channels, out_channels, h, w)
            vals=rng.standard_normal((3, 4, 3, 3)).astype(np.float32).flatten().tolist(),
        )
        B = onnx_helper.make_tensor(  # noqa: N806
            "B",
            onnx.TensorProto.FLOAT,
            [4],  # (out_channels,)
            vals=rng.standard_normal(4).astype(np.float32).tolist(),
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 4, 6, 6]
        )

        node = onnx_helper.make_node(
            "ConvTranspose",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )

        graph = onnx_helper.make_graph([node], "ConvTransposeModel", [X], [Y], [W, B])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Phase 5: Reduction and Utility Operations =====

    @staticmethod
    def create_clip_constant_bounds_model():
        """Create Clip ONNX model with constant min/max bounds.

        Test clamp operation with static bounds.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        min_val = onnx_helper.make_tensor("min_val", onnx.TensorProto.FLOAT, [1], vals=[0.0])
        max_val = onnx_helper.make_tensor("max_val", onnx.TensorProto.FLOAT, [1], vals=[1.0])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node("Clip", inputs=["X", "min_val", "max_val"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node], "ClipConstantBoundsModel", [X], [Y], [min_val, max_val]
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_clip_tensor_bounds_model():
        """Create Clip ONNX model with tensor bounds.

        Test clamp operation with tensor parameters for bounds.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        min_input = onnx_helper.make_tensor_value_info("min_input", onnx.TensorProto.FLOAT, [1])
        max_input = onnx_helper.make_tensor_value_info("max_input", onnx.TensorProto.FLOAT, [1])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node("Clip", inputs=["X", "min_input", "max_input"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node], "ClipTensorBoundsModel", [X, min_input, max_input], [Y]
        )

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_arange_literal_model():
        """Create Range (Arange) ONNX model with constant parameters.

        Test arange with literal start, stop, step values.

        :return: ONNX ModelProto
        """
        start = onnx_helper.make_tensor("start", onnx.TensorProto.INT64, [1], vals=[0])
        limit = onnx_helper.make_tensor("limit", onnx.TensorProto.INT64, [1], vals=[10])
        delta = onnx_helper.make_tensor("delta", onnx.TensorProto.INT64, [1], vals=[1])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.INT64, [10]
        )

        node = onnx_helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "ArangeLiteralModel", [], [Y], [start, limit, delta])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_arange_runtime_model():
        """Create Range (Arange) ONNX model with runtime parameters.

        Test arange with dynamic start, stop, step from inputs.

        :return: ONNX ModelProto
        """
        start = onnx_helper.make_tensor_value_info("start", onnx.TensorProto.INT64, [1])
        limit = onnx_helper.make_tensor_value_info("limit", onnx.TensorProto.INT64, [1])
        delta = onnx_helper.make_tensor_value_info("delta", onnx.TensorProto.INT64, [1])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.INT64, [None]
        )

        node = onnx_helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "ArangeRuntimeModel", [start, limit, delta], [Y])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_reshape_infer_dim_model():
        """Create Reshape ONNX model with dimension inference (-1).

        Test reshape with inferred dimension.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 20, 10]
        )
        shape_const = onnx_helper.make_tensor("shape", onnx.TensorProto.INT64, [2], vals=[200, -1])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [200, 1]
        )

        node = onnx_helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "ReshapeInferDimModel", [X], [Y], [shape_const])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    # ===== Phase 6: Simple Operations =====

    @staticmethod
    def create_squeeze_model():
        """Create Squeeze ONNX model with specific axis.

        Test squeeze with explicit dimension removal.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 1, 10, 1]
        )
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [2], vals=[1, 3])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node("Squeeze", inputs=["X", "axes"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "SqueezeModel", [X], [Y], [axes])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_unsqueeze_model():
        """Create Unsqueeze ONNX model.

        Test dimension insertion.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        axes = onnx_helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[2])
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10, 1]
        )

        node = onnx_helper.make_node("Unsqueeze", inputs=["X", "axes"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "UnsqueezeModel", [X], [Y], [axes])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_shape_model():
        """Create Shape ONNX model.

        Test shape extraction operation.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [2, 3, 4]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.INT64, [3]
        )

        node = onnx_helper.make_node("Shape", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "ShapeModel", [X], [Y])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_sign_model():
        """Create Sign ONNX model.

        Test unary sign operation.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node("Sign", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "SignModel", [X], [Y])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_trigonometric_model():
        """Create Sin ONNX model.

        Test trigonometric operation.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node("Sin", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "TrigonometricModel", [X], [Y])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model

    @staticmethod
    def create_floor_model():
        """Create Floor ONNX model.

        Test floor rounding operation.

        :return: ONNX ModelProto
        """
        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 10]
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 10]
        )

        node = onnx_helper.make_node("Floor", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph([node], "FloorModel", [X], [Y])

        model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
        model.ir_version = 8
        return model
