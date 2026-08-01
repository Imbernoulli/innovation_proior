__version__ = "2.8.3"

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "__version__",
]
