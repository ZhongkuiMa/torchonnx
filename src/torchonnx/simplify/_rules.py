"""Optimization rules for Stage 6 code refinement.

Defines which arguments are positional, which defaults to remove, etc.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "FUNCTION_DEFAULTS",
    "LAYER_DEFAULTS",
    "POSITIONAL_ONLY_ARGS",
]

# Layers where first N arguments should be positional (remove arg name)
# Format: "LayerType": ["arg1", "arg2", ...]
POSITIONAL_ONLY_ARGS: dict[str, list[str]] = {
    # Convolution layers
    "Conv1d": ["in_channels", "out_channels", "kernel_size"],
    "Conv2d": ["in_channels", "out_channels", "kernel_size"],
    "Conv3d": ["in_channels", "out_channels", "kernel_size"],
    "ConvTranspose1d": ["in_channels", "out_channels", "kernel_size"],
    "ConvTranspose2d": ["in_channels", "out_channels", "kernel_size"],
    "ConvTranspose3d": ["in_channels", "out_channels", "kernel_size"],
    # Linear layers
    "Linear": ["in_features", "out_features"],
    # Normalization layers
    "BatchNorm1d": ["num_features"],
    "BatchNorm2d": ["num_features"],
    "BatchNorm3d": ["num_features"],
    "LayerNorm": ["normalized_shape"],
    "GroupNorm": ["num_groups", "num_channels"],
    "InstanceNorm1d": ["num_features"],
    "InstanceNorm2d": ["num_features"],
    "InstanceNorm3d": ["num_features"],
    # Pooling layers
    "MaxPool1d": ["kernel_size"],
    "MaxPool2d": ["kernel_size"],
    "MaxPool3d": ["kernel_size"],
    "AvgPool1d": ["kernel_size"],
    "AvgPool2d": ["kernel_size"],
    "AvgPool3d": ["kernel_size"],
    "AdaptiveAvgPool1d": ["output_size"],
    "AdaptiveAvgPool2d": ["output_size"],
    "AdaptiveAvgPool3d": ["output_size"],
    "AdaptiveMaxPool1d": ["output_size"],
    "AdaptiveMaxPool2d": ["output_size"],
    "AdaptiveMaxPool3d": ["output_size"],
    # Activation layers
    "ReLU": [],
    "LeakyReLU": ["negative_slope"],
    "ELU": ["alpha"],
    "PReLU": [],
    "Sigmoid": [],
    "Tanh": [],
    "Softmax": ["dim"],
    "LogSoftmax": ["dim"],
    # Dropout layers
    "Dropout": ["p"],
    "Dropout2d": ["p"],
    "Dropout3d": ["p"],
    # Embedding layers
    "Embedding": ["num_embeddings", "embedding_dim"],
    # Recurrent layers
    "RNN": ["input_size", "hidden_size"],
    "LSTM": ["input_size", "hidden_size"],
    "GRU": ["input_size", "hidden_size"],
    # Shape operations
    "Flatten": ["start_dim"],
    "Unflatten": ["dim", "unflattened_size"],
    # Upsampling
    "Upsample": [],
    "UpsamplingNearest2d": [],
    "UpsamplingBilinear2d": [],
}

# Default argument values that can be omitted
# Format: "LayerType": {"arg_name": "default_value_str", ...}
LAYER_DEFAULTS: dict[str, dict[str, str]] = {
    # Convolution layers
    "Conv1d": {
        "stride": "1",
        "padding": "0",
        "dilation": "1",
        "groups": "1",
        "bias": "True",
        "padding_mode": "'zeros'",
    },
    "Conv2d": {
        "stride": "1",
        "padding": "0",
        "dilation": "1",
        "groups": "1",
        "bias": "True",
        "padding_mode": "'zeros'",
    },
    "ConvTranspose1d": {
        "stride": "1",
        "padding": "0",
        "output_padding": "0",
        "dilation": "1",
        "groups": "1",
        "bias": "True",
        "padding_mode": "'zeros'",
    },
    "ConvTranspose2d": {
        "stride": "1",
        "padding": "0",
        "output_padding": "0",
        "dilation": "1",
        "groups": "1",
        "bias": "True",
        "padding_mode": "'zeros'",
    },
    # Pooling layers
    "MaxPool2d": {
        "stride": "None",
        "padding": "0",
        "dilation": "1",
        "return_indices": "False",
        "ceil_mode": "False",
    },
    "AvgPool2d": {
        "stride": "None",
        "padding": "0",
        "ceil_mode": "False",
        "count_include_pad": "True",
        "divisor_override": "None",
    },
    "AdaptiveAvgPool2d": {},  # Only has output_size, which is required
    # Normalization layers
    "BatchNorm2d": {
        "eps": "1e-05",
        "momentum": "0.1",
        "affine": "True",
        "track_running_stats": "True",
    },
    # Linear layers
    "Linear": {
        "bias": "True",
    },
    # Activation functions
    "ReLU": {
        "inplace": "False",
    },
    "LeakyReLU": {
        "negative_slope": "0.01",
        "inplace": "False",
    },
    "ELU": {
        "alpha": "1.0",
        "inplace": "False",
    },
    "GELU": {
        "approximate": "'none'",
    },
    "Sigmoid": {},  # No parameters with defaults
    "Tanh": {},  # No parameters with defaults
    "Softmax": {},  # dim is required, in POSITIONAL_ONLY_ARGS
    # Dropout
    "Dropout": {
        "p": "0.5",
        "inplace": "False",
    },
    # Upsampling
    "Upsample": {
        "scale_factor": "None",
        "mode": "'nearest'",
        "align_corners": "None",
    },
    # Shape operations
    "Flatten": {
        "start_dim": "1",
        "end_dim": "-1",
    },
}

# Default argument values for functional operations (F.* and torch.*)
# Format: "function_name": {"arg_name": "default_value_str", ...}
FUNCTION_DEFAULTS: dict[str, dict[str, str]] = {
    # Activation functions (F.*)
    "F.relu": {
        "inplace": "False",
    },
    "F.leaky_relu": {
        "negative_slope": "0.01",
        "inplace": "False",
    },
    "F.elu": {
        "alpha": "1.0",
        "inplace": "False",
    },
    "F.gelu": {
        "approximate": "'none'",
    },
    "F.sigmoid": {},
    "F.tanh": {},
    "F.softmax": {
        "dtype": "None",
    },
    # Pooling functions (F.*)
    "F.max_pool2d": {
        "stride": "None",
        "padding": "0",
        "dilation": "1",
        "return_indices": "False",
        "ceil_mode": "False",
    },
    "F.avg_pool2d": {
        "stride": "None",
        "padding": "0",
        "ceil_mode": "False",
        "count_include_pad": "True",
        "divisor_override": "None",
    },
    "F.adaptive_avg_pool2d": {},
    # Convolution functions (F.*)
    "F.conv2d": {
        "stride": "1",
        "padding": "0",
        "dilation": "1",
        "groups": "1",
    },
    "F.conv_transpose2d": {
        "stride": "1",
        "padding": "0",
        "output_padding": "0",
        "groups": "1",
        "dilation": "1",
    },
    # Dropout functions (F.*)
    "F.dropout": {
        "p": "0.5",
        "training": "True",
        "inplace": "False",
    },
    # Torch tensor operations
    "torch.cat": {
        "dim": "0",
    },
    "torch.concat": {
        "dim": "0",
    },
    "torch.flatten": {
        "start_dim": "0",
        "end_dim": "-1",
    },
    "torch.reshape": {},
    "torch.squeeze": {
        "dim": "None",
    },
    "torch.unsqueeze": {},
    "torch.transpose": {},
    "torch.permute": {},
    # Torch math operations
    "torch.add": {
        "alpha": "1",
    },
    "torch.sub": {
        "alpha": "1",
    },
    "torch.mul": {},
    "torch.div": {},
    "torch.matmul": {},
    "torch.bmm": {},
    # Torch creation operations
    "torch.tensor": {
        "dtype": "None",
        "device": "None",
        "requires_grad": "False",
    },
    "torch.zeros": {
        "dtype": "None",
        "device": "None",
        "requires_grad": "False",
    },
    "torch.ones": {
        "dtype": "None",
        "device": "None",
        "requires_grad": "False",
    },
}
