import importlib
import os
import sys

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import ValueInfoProto


def reformat_io_shape(node: ValueInfoProto) -> list[int]:
    shape = [d.dim_value for d in node.type.tensor_type.shape.dim]
    if not len(shape) == 0:
        if shape[0] == 0:
            # Set the batch dimension to 1
            shape[0] = 1
        elif len(shape) > 1:
            # If there are multiple dimensions, we assume the first one is batch size
            # and set it to 1 if it is 0
            if shape[0] != 1:
                shape = [1] + shape

    return shape


def gen_module_name(file_path: str) -> str:
    file_path = os.path.normpath(file_path)
    module_name = file_path.split(f"{os.sep}")[-1].split(".")[0]
    # Remove all dots and dashes
    module_name = module_name.replace(".", "").replace("-", "")
    # Change to title case
    module_name = module_name.title().replace("_", "")
    # Remove all non-alphabetic and non-numeric characters
    module_name = "".join([c for c in module_name if c.isalpha() or c.isdigit()])
    # Remove the number at the beginning
    for i in range(len(module_name)):
        if module_name[i].isalpha():
            module_name = module_name[i:]
            break
    if module_name == "":
        raise ValueError(f"Cannot generate module name from {file_path}.")
    return module_name


def load_module_from_path(file_path: str, module_name: str = "custom_module"):
    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)  # noqa
    module = importlib.util.module_from_spec(spec)  # noqa
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compare_onnx_torch(
    onnx_path: str, torch_model_path: str, torch_params_path: str, atol: float = 1e-08
):
    # Use ONNX Runtime to load the ONNX model
    onnx_model = onnx.load(onnx_path)

    onnx_session = onnxruntime.InferenceSession(onnx_path)
    # Create a dummy input based on the ONNX model's input shape
    input_name = onnx_session.get_inputs()[0].name

    # Sometimes the model.graph.input contains initializers
    # We need to find the first input that is not an initializer
    initializers = dict(
        (initializer.name, initializer) for initializer in onnx_model.graph.initializer
    )
    input_node = None
    for input_node in onnx_model.graph.input:
        if input_node.name not in initializers:
            break
    assert input_node is not None

    input_shape = [d.dim_value for d in input_node.type.tensor_type.shape.dim]

    if input_shape[0] == 0:
        # Set the batch dimension to 1 if it is 0
        input_shape[0] = 1
    print(f"ONNX model input name: {input_name}")
    print(f"ONNX model input shape: {input_shape}")
    dummy_input_np = np.ones(input_shape, dtype=np.float32)
    onnx_output = onnx_session.run(None, {input_name: dummy_input_np})[0]

    # Import the torch model class file
    torch_model_name = gen_module_name(torch_model_path)
    torch_module = load_module_from_path(torch_model_path, torch_model_name)
    TempModule = getattr(torch_module, torch_model_name)
    torch_model = TempModule(dtype=torch.float64, params_path=torch_params_path)
    torch_model = torch_model.eval()
    input_shape = reformat_io_shape(input_node)
    print(f"Torch model input shape: {input_shape}")
    dummy_input = torch.ones(input_shape, dtype=torch.float64)
    torch_output = torch_model(dummy_input)
    # Compare the outputs

    onnx_output = torch.tensor(onnx_output).flatten().to(torch.float64)
    torch_output = torch_output.flatten()
    if not torch.allclose(onnx_output, torch_output, atol=atol):
        raise ValueError(
            f"Outputs do not match for {onnx_path} and {torch_model_path}.\n"
            f"ONNX output: {onnx_output[:100].tolist()}\n"
            f"Torch output: {torch_output[:100].tolist()}\n"
            f"Max diff: {torch.max(torch.abs(onnx_output - torch_output)):.6f}\n"
        )


def check_torch_model(target_dir_path: str, atol: float = 1e-08):
    # Read the original ONNX model in "onnx" subdirectory.
    onnx_dir = os.path.join(target_dir_path, "onnx_slim")
    torch_dir = os.path.join(target_dir_path, "onnx_torch")

    # Iterate each onnx model in the directory.
    # There exists a corresponding torch model in the "onnx_torch" subdirectory.
    for onnx_file in os.listdir(onnx_dir):
        print("Checking ONNX model:", onnx_file)
        if not onnx_file.endswith(".onnx"):
            continue

        onnx_path = os.path.join(onnx_dir, onnx_file)
        torch_model_path = os.path.join(torch_dir, onnx_file.replace(".onnx", ".py"))
        torch_params_path = os.path.join(torch_dir, onnx_file.replace(".onnx", ".pth"))

        if not os.path.exists(torch_model_path):
            raise FileNotFoundError(
                f"Corresponding torch model not found for {onnx_file} at {torch_model_path}"
            )
        if not os.path.exists(torch_params_path):
            raise FileNotFoundError(
                f"Corresponding torch parameters not found for {onnx_file} at {torch_params_path}"
            )

        onnx_path = onnx_path.replace("_slimmed", "").replace("_slim", "")
        compare_onnx_torch(onnx_path, torch_model_path, torch_params_path, atol=atol)


if __name__ == "__main__":
    # numerical issues
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/cora"
    check_torch_model(target_dir_path, atol=1e-04)
    target_dir_path = "../../vnncomp2025_benchmarks/benchmarks/malbeware"
    check_torch_model(target_dir_path, atol=1e-05)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/acasxu_2023"
    check_torch_model(target_dir_path, atol=1e-06)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/metaroom_2023"
    check_torch_model(target_dir_path, atol=1e-04)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/linearizenn"
    check_torch_model(target_dir_path, atol=1e-05)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/yolo_2023"
    check_torch_model(target_dir_path, atol=1e-05)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/vggnet16_2023"  #
    check_torch_model(target_dir_path, atol=1e-05)
    target_dir_path = "../../vnncomp2025_benchmarks/benchmarks/relusplitter"
    check_torch_model(target_dir_path, atol=1e-04)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/dist_shift_2023"
    check_torch_model(target_dir_path, atol=1e-05)
    target_dir_path = (
        "../../vnncomp2024_benchmarks/benchmarks/collins_aerospace_benchmark"
    )
    check_torch_model(target_dir_path, atol=1e-03)
    target_dir_path = "../../vnncomp2025_benchmarks/benchmarks/lsnc_relu"
    check_torch_model(target_dir_path, atol=1e-06)

    target_dir_path = "../../vnncomp2025_benchmarks/benchmarks/test"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2025_benchmarks/benchmarks/sat_relu"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/safenlp"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2025_benchmarks/benchmarks/cersyve"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/tllverifybench_2023"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/collins_rul_cnn_2023"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/cifar100"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/tinyimagenet"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2025_benchmarks/benchmarks/soundnessbench"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/vit_2023"
    check_torch_model(target_dir_path)
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/lsnc"
    check_torch_model(target_dir_path)

    # We need real inputs for the following benchmarks.
    # target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/cgan_2023"
    # target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/cctsdb_yolo_2023"
    target_dir_path = "../../vnncomp2024_benchmarks/benchmarks/ml4acopf_2024"

    # NOT SUPPORTED BINARY NETWORKS
    # target_dir_path = (
    #     "../../vnncomp2024_benchmarks/benchmarks/traffic_signs_recognition_2023"
    # )
