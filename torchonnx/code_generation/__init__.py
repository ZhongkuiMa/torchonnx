"""PyTorch code generation from intermediate representation.

This package contains modules for generating PyTorch code from IR:
- init_gen: Generate __init__ method
- forward_gen: Generate forward() method
- module_gen: Generate complete module code
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_pytorch_module_with_state_dict"]

from .module_gen import generate_pytorch_module_with_state_dict
