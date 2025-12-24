"""ONNX model preprocessing utilities."""

__docformat__ = "restructuredtext"
__all__ = ["load_and_preprocess_onnx_model"]

import warnings

import onnx
from onnx import ModelProto, TensorProto, version_converter

# Recommended opset range based on shapeonnx compatibility testing
RECOMMENDED_OPSET = 20
MIN_TESTED_OPSET = 17  # Lowest working version for maximum compatibility
MAX_TESTED_OPSET = 21  # Highest tested working version
SLIMONNX_VERSION = "1.0.0"


def _clear_docstrings(model: ModelProto) -> ModelProto:
    """Clear all docstrings from ONNX model nodes.

    :param model: Input ONNX model
    :return: Model with cleared docstrings
    """
    for node in model.graph.node:
        node.doc_string = ""
    return model


def _check_model(model: ModelProto) -> None:
    """Check ONNX model validity using onnx.checker.

    :param model: Input ONNX model
    :raises ValueError: If model is invalid
    """
    try:
        onnx.checker.check_model(model)
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(f"Invalid ONNX model: {error}") from error


def _convert_version(
    model: ModelProto,
    target_opset: int = RECOMMENDED_OPSET,
    warn_on_diff: bool = True,
) -> ModelProto:
    """Convert ONNX model to specified opset version.

    :param model: Input ONNX model
    :param target_opset: Target opset version
    :param warn_on_diff: Warn if target differs from recommended
    :return: Converted model (IR version set automatically by ONNX)
    """
    current_opset = model.opset_import[0].version if model.opset_import else 0

    # Warn if target_opset is outside recommended range
    if warn_on_diff and not (MIN_TESTED_OPSET <= target_opset <= MAX_TESTED_OPSET):
        warnings.warn(
            f"Target opset {target_opset} is outside "
            f"tested range [{MIN_TESTED_OPSET}, {MAX_TESTED_OPSET}]. "
            f"Recommended opset is {RECOMMENDED_OPSET} for maximum compatibility.",
            UserWarning,
            stacklevel=2,
        )

    # Convert if different
    if current_opset != target_opset:
        try:
            model = version_converter.convert_version(model, target_opset)
        except (ValueError, RuntimeError, AttributeError) as error:
            warnings.warn(
                f"Version conversion failed "
                f"from opset {current_opset} to {target_opset}: {error}. "
                f"Keeping original opset version.",
                UserWarning,
                stacklevel=2,
            )

    return model


def _apply_shapeonnx_inference(model: ModelProto, shapes: dict) -> None:
    """Apply inferred shapes from shapeonnx to model value_info.

    :param model: ONNX model to update
    :param shapes: Dictionary of inferred shapes from shapeonnx
    """
    for node in model.graph.node:
        for output in node.output:
            if output in shapes:
                found = False
                for value_info in model.graph.value_info:
                    if value_info.name == output:
                        shape_val = shapes[output]
                        shape_list = [shape_val] if isinstance(shape_val, int) else shape_val
                        value_info.type.tensor_type.shape.ClearField("dim")
                        for dim_value in shape_list:
                            dim = value_info.type.tensor_type.shape.dim.add()
                            dim.dim_value = dim_value
                        found = True
                        break

                if not found:
                    value_info = model.graph.value_info.add()
                    value_info.name = output
                    shape_val = shapes[output]
                    shape_list = [shape_val] if isinstance(shape_val, int) else shape_val
                    for dim_value in shape_list:
                        dim = value_info.type.tensor_type.shape.dim.add()
                        dim.dim_value = dim_value


def _infer_shapes(model: ModelProto, use_shapeonnx: bool = False) -> ModelProto:
    """Run shape inference with error handling.

    :param model: Input ONNX model
    :param use_shapeonnx: Use shapeonnx library instead of ONNX's built-in inference
    :return: Model with inferred shapes (if successful)
    """
    if use_shapeonnx:
        try:
            from shapeonnx import infer_onnx_shape

            input_nodes = list(model.graph.input)
            output_nodes = list(model.graph.output)
            nodes = list(model.graph.node)
            initializers = {init.name: init for init in model.graph.initializer}

            shapes = infer_onnx_shape(
                input_nodes=input_nodes,
                output_nodes=output_nodes,
                nodes=nodes,
                initializers=initializers,
                has_batch_dim=True,
                verbose=False,
            )
            _apply_shapeonnx_inference(model, shapes)
        except (ValueError, RuntimeError, AttributeError, ImportError) as error:
            warnings.warn(
                f"ShapeONNX inference failed: {error}. Falling back to ONNX inference.",
                UserWarning,
                stacklevel=2,
            )
            try:
                model = onnx.shape_inference.infer_shapes(model)
            except (ValueError, RuntimeError, AttributeError) as error2:
                warnings.warn(
                    f"ONNX shape inference also failed: {error2}",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except (ValueError, RuntimeError, AttributeError) as error:
            warnings.warn(f"Shape inference failed: {error}", UserWarning, stacklevel=2)
    return model


def _convert_onnx_constants_to_initializers(model: ModelProto) -> ModelProto:
    """Convert Constant nodes to initializers.

    ONNX Constant nodes contain constant tensors that don't change during inference.
    This function extracts these constants and adds them as initializers, then
    removes the Constant nodes from the graph.

    :param model: Input ONNX model
    :return: Modified ONNX model with Constant nodes converted to initializers
    """
    new_initializers = []
    nodes_to_remove = []

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue

        if not node.output:
            continue

        output_name = node.output[0]

        for attr in node.attribute:
            if attr.name == "value":
                tensor = attr.t
                new_tensor = TensorProto()
                new_tensor.CopyFrom(tensor)
                new_tensor.name = output_name

                new_initializers.append(new_tensor)
                nodes_to_remove.append(node)
                break

    if not new_initializers:
        return model

    model_copy = ModelProto()
    model_copy.CopyFrom(model)

    for tensor in new_initializers:
        model_copy.graph.initializer.append(tensor)

    remaining_nodes = [n for n in model_copy.graph.node if n not in nodes_to_remove]

    del model_copy.graph.node[:]
    model_copy.graph.node.extend(remaining_nodes)

    return model_copy


def load_and_preprocess_onnx_model(
    onnx_path: str,
    target_opset: int | None = None,
    infer_shapes: bool = True,
    check_model: bool = True,
    clear_docstrings: bool = True,
    eliminate_constants: bool = True,
    use_shapeonnx: bool = False,
) -> ModelProto:
    """Load ONNX model and preprocess for SlimONNX.

    Preprocessing steps:
    1. Load model from file
    2. Validate with ONNX checker (if enabled)
    3. Convert to target opset version (if specified)
    4. Run shape inference (if enabled)
    5. Convert Constant nodes to initializers (if enabled)
    6. Clear node docstrings (if enabled)

    :param onnx_path: Path to ONNX file
    :param target_opset: Target opset version (None = keep original)
    :param infer_shapes: Whether to run shape inference
    :param check_model: Whether to validate model with onnx.checker
    :param clear_docstrings: Whether to clear node docstrings
    :param eliminate_constants: Whether to convert Constant nodes to initializers
    :param use_shapeonnx: Use shapeonnx library for shape inference instead of ONNX
    :return: Preprocessed model
    """
    model = onnx.load(onnx_path)

    if check_model:
        _check_model(model)

    if target_opset is not None:
        model = _convert_version(model, target_opset=target_opset)

    if check_model:
        _check_model(model)

    if infer_shapes:
        model = _infer_shapes(model, use_shapeonnx=use_shapeonnx)

    if eliminate_constants:
        model = _convert_onnx_constants_to_initializers(model)

    if clear_docstrings:
        model = _clear_docstrings(model)

    return model
