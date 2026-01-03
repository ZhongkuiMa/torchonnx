"""Analyze ONNX model nodes and compare with PyTorch module outputs.

This tool runs an ONNX model node-by-node using onnxruntime and compares
the intermediate outputs with the corresponding PyTorch module to identify
discrepancies.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, "..")


def load_pytorch_module(module_path: str, state_dict_path: str):
    """Load PyTorch module and state dict.

    :param module_path: Path to PyTorch module file
    :param state_dict_path: Path to state dict file
    :return: Loaded PyTorch model
    """
    module_file = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_file.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_name = module.__all__[0]
    model_class = getattr(module, class_name)
    model = model_class()

    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def run_onnx_node_by_node(onnx_model_path: str, input_data: np.ndarray):
    """Run ONNX model and capture all intermediate outputs.

    :param onnx_model_path: Path to ONNX model
    :param input_data: Input data as numpy array
    :return: Dictionary mapping tensor names to their values
    """
    # Load ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Create session
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run full model to get all outputs
    # We'll need to modify the model to output intermediate values
    outputs = session.run(None, {input_name: input_data})

    # Get output names
    output_names = [out.name for out in session.get_outputs()]

    print(f"ONNX Model: {onnx_model_path}")
    print(f"Input: {input_name}, shape: {input_data.shape}")
    print(f"Outputs: {output_names}")
    print(f"Number of nodes: {len(onnx_model.graph.node)}")

    return dict(zip(output_names, outputs, strict=False))


def create_intermediate_onnx_model(onnx_model_path: str, output_names: list):
    """Create a modified ONNX model that outputs intermediate tensors.

    :param onnx_model_path: Path to original ONNX model
    :param output_names: List of intermediate tensor names to output
    :return: Modified ONNX model
    """
    onnx_model = onnx.load(onnx_model_path)

    # Add intermediate outputs to the graph
    for name in output_names:
        # Find the value_info for this tensor
        value_info = None
        for vi in onnx_model.graph.value_info:
            if vi.name == name:
                value_info = vi
                break

        if value_info is None:
            # Create a new value_info
            value_info = onnx_model.graph.value_info.add()
            value_info.name = name

        # Check if already an output
        is_output = any(out.name == name for out in onnx_model.graph.output)
        if not is_output:
            # Add to outputs
            output = onnx_model.graph.output.add()
            output.name = name
            output.type.CopyFrom(value_info.type)

    return onnx_model


def _analyze_pytorch_model(pytorch_module_path: str, state_dict_path: str, input_data: np.ndarray):
    """Load and run PyTorch model.

    :return: (pytorch_model, pytorch_output) or (None, None) on error
    """
    print("Loading PyTorch model...")
    try:
        pytorch_model = load_pytorch_module(pytorch_module_path, state_dict_path)
        print("[OK] PyTorch model loaded successfully\n")
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"[FAIL] Failed to load PyTorch model: {e}\n")
        return None, None

    print("Running PyTorch model...")
    try:
        with torch.no_grad():
            pytorch_input = torch.from_numpy(input_data)
            pytorch_output = pytorch_model(pytorch_input)

        if isinstance(pytorch_output, torch.Tensor):
            pytorch_output = pytorch_output.numpy()
        print(f"[OK] PyTorch output shape: {pytorch_output.shape}\n")
        return pytorch_model, pytorch_output
    except (RuntimeError, ValueError, TypeError, AttributeError) as e:
        print(f"[FAIL] PyTorch model failed: {e}\n")
        import traceback

        traceback.print_exc()
        return None, None


def _run_onnx_with_intermediates(
    onnx_model_path: str,
    onnx_model,
    intermediate_names: list,
    input_data: np.ndarray,
):
    """Run ONNX model and capture intermediate outputs.

    :return: (onnx_results dict) or empty dict on error
    """
    try:
        modified_model = create_intermediate_onnx_model(onnx_model_path, intermediate_names)
        temp_model_path = Path(onnx_model_path).parent / "temp_intermediate.onnx"
        onnx.save(modified_model, str(temp_model_path))

        session = ort.InferenceSession(str(temp_model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        onnx_outputs = session.run(None, {input_name: input_data})
        output_names = [out.name for out in session.get_outputs()]
        onnx_results = dict(zip(output_names, onnx_outputs, strict=False))

        temp_model_path.unlink()
        print(f"[OK] ONNX model executed, captured {len(onnx_results)} outputs\n")
        return onnx_results
    except (RuntimeError, ValueError, OSError) as e:
        print(f"[FAIL] Failed to run ONNX with intermediate outputs: {e}\n")
        # Fall back to just final output
        session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        onnx_outputs = session.run(None, {input_name: input_data})
        output_names = [out.name for out in session.get_outputs()]
        onnx_results = dict(zip(output_names, onnx_outputs, strict=False))
        print(f"Using final outputs only: {len(onnx_results)} outputs\n")
        return onnx_results


def _print_node_analysis(onnx_model, onnx_results: dict, max_nodes: int | None):
    """Print analysis of ONNX model nodes.

    :param onnx_model: ONNX model
    :param onnx_results: Dictionary of captured outputs
    :param max_nodes: Maximum nodes to analyze
    """
    print(f"\n{'=' * 80}")
    print("Node Analysis")
    print(f"{'=' * 80}\n")

    nodes_to_analyze = onnx_model.graph.node[:max_nodes] if max_nodes else onnx_model.graph.node

    for idx, node in enumerate(nodes_to_analyze):
        print(f"Node {idx + 1}/{len(nodes_to_analyze)}: {node.op_type}")
        print(f"  Name: {node.name}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")

        for output_name in node.output:
            if output_name in onnx_results:
                output_val = onnx_results[output_name]
                print(
                    f"  Output '{output_name}': shape={output_val.shape}, "
                    f"dtype={output_val.dtype}, "
                    f"range=[{output_val.min():.6f}, {output_val.max():.6f}]"
                )
            else:
                print(f"  Output '{output_name}': not captured")

        print()


def _print_final_output_comparison(onnx_model, onnx_results: dict, pytorch_output: np.ndarray):
    """Print comparison of final outputs.

    :param onnx_model: ONNX model
    :param onnx_results: Dictionary of captured outputs
    :param pytorch_output: PyTorch output
    """
    print(f"\n{'=' * 80}")
    print("Final Output Comparison")
    print(f"{'=' * 80}\n")

    final_output_name = onnx_model.graph.output[0].name
    if final_output_name not in onnx_results:
        print(f"[FAIL] Final output '{final_output_name}' not found in ONNX results")
        return

    onnx_final = onnx_results[final_output_name]
    print(f"ONNX output shape: {onnx_final.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")

    if onnx_final.shape != pytorch_output.shape:
        print("\n[FAIL] Shape mismatch!")
        return

    diff = np.abs(onnx_final - pytorch_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print("\n[OK] Shapes match!")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("[OK] Outputs match within tolerance!")
    else:
        print("[WARN] Outputs differ significantly")
        onnx_min, onnx_max = onnx_final.min(), onnx_final.max()
        print(f"ONNX output range: [{onnx_min:.6f}, {onnx_max:.6f}]")
        pt_min, pt_max = pytorch_output.min(), pytorch_output.max()
        print(f"PyTorch output range: [{pt_min:.6f}, {pt_max:.6f}]")


def analyze_model_nodes(
    onnx_model_path: str,
    pytorch_module_path: str,
    state_dict_path: str,
    input_shape: tuple | None = None,
    max_nodes: int | None = None,
):
    """Analyze ONNX model nodes and compare with PyTorch.

    :param onnx_model_path: Path to ONNX model
    :param pytorch_module_path: Path to PyTorch module
    :param state_dict_path: Path to state dict
    :param input_shape: Optional input shape (auto-detected if None)
    :param max_nodes: Maximum number of nodes to analyze (None for all)
    """
    # Load ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Get input shape
    if input_shape is None:
        input_tensor = onnx_model.graph.input[0]
        input_shape = tuple(
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in input_tensor.type.tensor_type.shape.dim
        )

    print(f"\n{'=' * 80}")
    print("Analyzing ONNX Model Node-by-Node")
    print(f"{'=' * 80}")
    print(f"ONNX Model: {onnx_model_path}")
    print(f"PyTorch Module: {pytorch_module_path}")
    print(f"Input shape: {input_shape}")
    print(f"{'=' * 80}\n")

    # Create input data
    rng = np.random.default_rng()
    input_data = rng.standard_normal(input_shape).astype(np.float32)

    # Load and run PyTorch model
    _pytorch_model, pytorch_output = _analyze_pytorch_model(
        pytorch_module_path, state_dict_path, input_data
    )
    if pytorch_output is None:
        return

    # Run ONNX model with all intermediate outputs
    print("Running ONNX model...")

    # Collect all intermediate tensor names
    intermediate_names = []
    nodes = onnx_model.graph.node[:max_nodes] if max_nodes else onnx_model.graph.node
    for node in nodes:
        for output in node.output:
            if output not in intermediate_names:
                intermediate_names.append(output)

    print(f"Found {len(intermediate_names)} intermediate tensors\n")

    onnx_results = _run_onnx_with_intermediates(
        onnx_model_path, onnx_model, intermediate_names, input_data
    )

    # Analyze nodes
    _print_node_analysis(onnx_model, onnx_results, max_nodes)

    # Compare final outputs
    _print_final_output_comparison(onnx_model, onnx_results, pytorch_output)

    print(f"\n{'=' * 80}\n")


def main():
    """Parse arguments and analyze ONNX model nodes."""
    parser = argparse.ArgumentParser(
        description="Analyze ONNX model nodes and compare with PyTorch module"
    )
    parser.add_argument("onnx_model", type=str, help="Path to ONNX model file")
    parser.add_argument("pytorch_module", type=str, help="Path to PyTorch module file (.py)")
    parser.add_argument("state_dict", type=str, help="Path to state dict file (.pth)")
    parser.add_argument(
        "--input-shape",
        type=str,
        help="Input shape as comma-separated values (e.g., '1,3,224,224')",
    )
    parser.add_argument("--max-nodes", type=int, help="Maximum number of nodes to analyze")

    args = parser.parse_args()

    # Parse input shape
    input_shape = None
    if args.input_shape:
        input_shape = tuple(int(x) for x in args.input_shape.split(","))

    analyze_model_nodes(
        args.onnx_model,
        args.pytorch_module,
        args.state_dict,
        input_shape=input_shape,
        max_nodes=args.max_nodes,
    )


if __name__ == "__main__":
    main()
