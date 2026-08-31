"""ONNX model preprocessing utilities."""

__docformat__ = "restructuredtext"
__all__ = ["load_and_preprocess_onnx_model", "preprocess_onnx_model"]

import warnings
from typing import Any

import onnx
from onnx import ModelProto, TensorProto, version_converter

# Recommended opset range based on shapeonnx compatibility testing
RECOMMENDED_OPSET = 20
MIN_TESTED_OPSET = 17  # Lowest working version for maximum compatibility
MAX_TESTED_OPSET = 21  # Highest tested working version
SLIMONNX_VERSION = "1.0.0"


def _canonicalize_opset_imports(model: ModelProto) -> ModelProto:
    """Return a model with at most one import per ONNX domain.

    Some valid models contain repeated, identical ``opset_import`` entries.
    ONNX's native version converter updates only one duplicate and leaves a
    semantically inconsistent model behind.  Identical duplicates carry no
    information and are collapsed; conflicting versions are ambiguous and
    therefore rejected before conversion.

    :param model: Input ONNX model. The input is not mutated.
    :return: Model with canonical opset imports.
    :raises ValueError: If one domain declares conflicting versions.
    """
    imports: list[tuple[str, int]] = []
    versions: dict[str, int] = {}
    has_duplicates = False
    for opset in model.opset_import:
        domain = opset.domain
        version = opset.version
        if domain not in versions:
            versions[domain] = version
            imports.append((domain, version))
            continue
        previous = versions[domain]
        if previous != version:
            label = domain or "ai.onnx"
            raise ValueError(
                f"Conflicting opset imports for domain {label!r}: {previous} and {version}"
            )
        has_duplicates = True

    if not has_duplicates:
        return model

    canonical = ModelProto()
    canonical.CopyFrom(model)
    del canonical.opset_import[:]
    for domain, version in imports:
        opset = canonical.opset_import.add()
        opset.domain = domain
        opset.version = version
    return canonical


def _primary_opset(model: ModelProto) -> int:
    """Return the default ONNX-domain opset version."""
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    raise ValueError("Model has no primary ONNX opset import")


def _clear_docstrings(model: ModelProto) -> ModelProto:
    """Clear all docstrings from ONNX model nodes.

    :param model: Input ONNX model.

    :return: Model with cleared docstrings
    """
    for node in model.graph.node:
        node.doc_string = ""
    return model


def _check_model(model: ModelProto) -> None:
    """Check ONNX model validity using onnx.checker.

    :param model: Input ONNX model.

    :raises ValueError: If model is invalid.

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

    :param model: Input ONNX model.
    :param target_opset: Target opset version.
    :param warn_on_diff: Warn if target differs from recommended.

    :return: Converted model (IR version set automatically by ONNX)
    """
    model = _canonicalize_opset_imports(model)
    current_opset = _primary_opset(model)

    # A zero-node graph is an identity alias and has no versioned operator
    # semantics. Native ONNX conversion is both unnecessary and unable to
    # represent a rewrite for this valid degenerate graph.
    if not model.graph.node:
        return model

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
            raise ValueError(
                f"Failed to convert model from opset {current_opset} to {target_opset}: {error}. "
                f"Use target_opset={current_opset} to keep the original opset."
            ) from error

    return model


def _apply_shapeonnx_inference(model: ModelProto, shapes: dict) -> None:
    value_info_map: dict[str, Any] = {}
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi

    for node in model.graph.node:
        for output in node.output:
            if output not in shapes:
                continue
            vi = value_info_map.get(output)
            shape_val = shapes[output]
            shape_list = [shape_val] if isinstance(shape_val, int) else shape_val
            if vi is not None:
                vi.type.tensor_type.shape.ClearField("dim")
                for dim_value in shape_list:
                    dim = vi.type.tensor_type.shape.dim.add()
                    dim.dim_value = dim_value
            else:
                vi = model.graph.value_info.add()
                vi.name = output
                for dim_value in shape_list:
                    dim = vi.type.tensor_type.shape.dim.add()
                    dim.dim_value = dim_value
                value_info_map[output] = vi


def _infer_shapes(model: ModelProto, use_shapeonnx: bool = False) -> ModelProto:
    """Run shape inference with error handling.

    :param model: Input ONNX model.
    :param use_shapeonnx: Use shapeonnx library instead of ONNX's built-in inference.

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

    :param model: Input ONNX model.

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

    # Use output-name matching, not object identity, because CopyFrom
    # creates new Python objects and the identity-based ``in`` would
    # never match a single node.
    removed_outputs = {n.output[0] for n in nodes_to_remove if n.output}
    remaining_nodes = [n for n in model_copy.graph.node if n.output[0] not in removed_outputs]

    del model_copy.graph.node[:]
    model_copy.graph.node.extend(remaining_nodes)

    return model_copy


def preprocess_onnx_model(
    model: ModelProto,
    *,
    target_opset: int | None = None,
    infer_shapes: bool = True,
    check_model: bool = True,
    clear_docstrings: bool = True,
    eliminate_constants: bool = True,
    use_shapeonnx: bool = False,
) -> ModelProto:
    """Normalize an in-memory ONNX model for IR construction.

    Preprocessing steps:
    1. Copy the caller-owned model
    2. Validate with ONNX checker (if enabled)
    3. Convert to target opset version (if specified)
    4. Run shape inference (if enabled)
    5. Convert Constant nodes to initializers (if enabled)
    6. Clear node docstrings (if enabled)

    :param model: Caller-owned ONNX model. The input is not mutated.
    :param target_opset: Target opset version (None = keep original).
    :param infer_shapes: Whether to run shape inference.
    :param check_model: Whether to validate model with onnx.checker.
    :param clear_docstrings: Whether to clear node docstrings.
    :param eliminate_constants: Whether to convert Constant nodes to initializers.
    :param use_shapeonnx: Use shapeonnx library for shape inference instead of ONNX.

    :return: Preprocessed model
    """
    normalized = ModelProto()
    normalized.CopyFrom(model)
    normalized = _canonicalize_opset_imports(normalized)

    if check_model:
        _check_model(normalized)

    if target_opset is not None:
        normalized = _convert_version(normalized, target_opset=target_opset)

    if infer_shapes:
        normalized = _infer_shapes(normalized, use_shapeonnx=use_shapeonnx)

    if eliminate_constants:
        normalized = _convert_onnx_constants_to_initializers(normalized)

    if clear_docstrings:
        normalized = _clear_docstrings(normalized)

    return normalized


def load_and_preprocess_onnx_model(
    onnx_path: str,
    target_opset: int | None = None,
    infer_shapes: bool = True,
    check_model: bool = True,
    clear_docstrings: bool = True,
    eliminate_constants: bool = True,
    use_shapeonnx: bool = False,
) -> ModelProto:
    """Load an ONNX file and normalize it for IR construction.

    :param onnx_path: Path to ONNX file.
    :param target_opset: Target opset version (None = keep original).
    :param infer_shapes: Whether to run shape inference.
    :param check_model: Whether to validate model with onnx.checker.
    :param clear_docstrings: Whether to clear node docstrings.
    :param eliminate_constants: Whether to convert Constant nodes to initializers.
    :param use_shapeonnx: Use shapeonnx library for shape inference instead of ONNX.
    :return: Preprocessed model.
    """
    return preprocess_onnx_model(
        onnx.load(onnx_path),
        target_opset=target_opset,
        infer_shapes=infer_shapes,
        check_model=check_model,
        clear_docstrings=clear_docstrings,
        eliminate_constants=eliminate_constants,
        use_shapeonnx=use_shapeonnx,
    )
