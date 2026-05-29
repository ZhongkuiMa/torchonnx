"""Runtime helpers inlined into generated PyTorch modules.

Each submodule defines real Python helper functions (``dynamic_slice``,
``scatter_nd``, ``dynamic_expand``) so that ruff, mypy, and pytest can
see them. The code generator extracts the source via
``inspect.getsource()`` at emission time and pastes it into the
generated file; the runtime semantics of the inlined copy are therefore
exactly what the unit suite tests here.

Two parallel implementations live in ``_standard`` and ``_vmap``:

* ``_standard`` uses ``torch.index_put_`` / ``slice`` / ``data.expand``
  for the simplest possible runtime. ``dynamic_slice`` returns just the
  sliced tensor.
* ``_vmap`` uses functional ``torch.scatter`` / ``torch.gather`` so the
  helpers compose with ``torch.vmap`` and ``torch.compile``.
  ``dynamic_slice`` returns ``(result, valid_flag)`` so a downstream
  ``scatter_nd`` can know whether the slice was empty without Python
  branching.

The two share function names because the generated module only ever
contains one set; the code generator picks per ``vmap_mode``.
"""

__docformat__ = "restructuredtext"
__all__: list[str] = []
