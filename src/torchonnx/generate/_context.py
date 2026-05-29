"""Forward-generation context shared between code_generator and op handlers.

Lifted out of ``_forward_gen.py`` so per-handler files (``_handlers/_*.py``)
can import the accessor at module top instead of doing a lazy
``from torchonnx.generate._forward_gen import _get_ctx`` inside every
function body. The previous arrangement was forced by a circular import
(``_forward_gen`` -> ``_handlers`` -> ``_forward_gen``); routing the
context through this leaf module breaks the cycle.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ForwardGenContext",
    "_get_ctx",
    "get_forward_gen_context",
    "set_forward_gen_context",
]

import threading


class ForwardGenContext:
    """Per-generation accumulator and analysis cache for forward-method emission.

    Carries four roles today (all single-threaded, single-conversion):

    1. usage accumulator -- which constants / parameters / helpers actually
       got referenced, so unused buffers can be pruned;
    2. precomputed analysis -- slice-length hints for vmap mode;
    3. emission config -- ``vmap_mode`` flag, ``first_input_name`` for
       device inference;
    4. producer linkage -- ``slice_valid_var_by_output`` so a directly
       downstream ScatterND can consume the validity flag of the specific
       Slice that produced its data tensor.

    Splitting these into four orthogonal objects is a deeper refactor;
    this module currently bundles them under one type so the migration
    is mechanical.
    """

    def __init__(self):
        self.used_constants: set[str] = set()
        self.used_parameters: set[str] = set()
        # Helper-function need flags, populated by IR pre-analysis.
        self.needs_dynamic_slice: bool = False
        self.needs_scatter_nd: bool = False
        self.needs_dynamic_expand: bool = False
        # First input name for device inference (e.g., "x0").
        self.first_input_name: str | None = None
        # Vmap mode settings.
        self.vmap_mode: bool = True
        # Pre-computed slice lengths for vmap mode: {layer_name: [lengths]}.
        # When provided, dynamic_slice uses these instead of computing with .item().
        self.slice_length_hints: dict[str, list[int]] = {}
        # Per-slice validity variable: {output_code_name: valid_var_name}.
        # Lets a downstream ScatterND consume the validity flag of the specific
        # Slice that produced its data tensor, instead of multiplying every
        # slice's validity into a global accumulator (which caused one
        # out-of-bounds slice anywhere to silently zero every later scatter).
        self.slice_valid_var_by_output: dict[str, str] = {}

    def mark_constant_used(self, constant_name: str) -> None:
        """Mark a constant as used in forward method."""
        self.used_constants.add(constant_name)

    def mark_parameter_used(self, parameter_name: str) -> None:
        """Mark a parameter as used in forward method."""
        self.used_parameters.add(parameter_name)

    def get_slice_lengths(self, layer_name: str) -> list[int] | None:
        """Get pre-computed slice lengths for a Slice layer (vmap mode)."""
        return self.slice_length_hints.get(layer_name)


# Thread-local context store so two concurrent ``TorchONNX().convert(...)``
# calls (or any other parallel use of ``generate_pytorch_module``) do not
# stomp on each other's accumulator state. The earlier module-level
# singleton was the dian F2 / pianzhi #8 reentrancy hazard called out in
# the R6 / R10 audits: one thread's ScatterND would read another thread's
# ``slice_valid_var_by_output`` and emit a generated module that referenced
# variables it had never produced.
_thread_local = threading.local()


def get_forward_gen_context() -> ForwardGenContext | None:
    """Get the current thread's forward generation context (or None)."""
    return getattr(_thread_local, "ctx", None)


def set_forward_gen_context(context: ForwardGenContext | None) -> None:
    """Set the current thread's forward generation context.

    :param context: Context to set, or None to clear.
    """
    _thread_local.ctx = context


def _get_ctx() -> ForwardGenContext:
    """Get the current thread's context, asserting it is set.

    Used from handlers that expect the context to be initialised.

    :return: Current forward generation context.
    :raises RuntimeError: If context is not initialized on this thread.
    """
    ctx = getattr(_thread_local, "ctx", None)
    if ctx is None:
        raise RuntimeError(
            "ForwardGenContext not initialized. "
            "Set context via set_forward_gen_context() before calling handlers."
        )
    assert isinstance(ctx, ForwardGenContext)
    return ctx
