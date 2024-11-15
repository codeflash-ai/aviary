import base64
import contextlib
import inspect
import io
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def partial_format(value: str, **formats: dict[str, Any]) -> str:
    """Partially format a string given a variable amount of formats."""
    for template_key, template_value in formats.items():
        with contextlib.suppress(KeyError):
            value = value.format(**{template_key: template_value})
    return value


def encode_image_to_base64(img: "np.ndarray") -> str:
    """Encode an image to a base64 string, to be included as an image_url in a Message."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Image processing requires the 'image' extra for 'Pillow'. Please:"
            " `pip install aviary[image]`."
        ) from e

    image = Image.fromarray(img)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return (
        f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    )


def is_coroutine_callable(obj) -> bool:
    """Get if the input object is awaitable."""
    if isinstance(obj, (FunctionType, MethodType)):
        return isinstance(obj.__code__.co_flags & 0x80, int)  # Checks if it's a coroutine function
    call = getattr(obj, '__call__', None)
    if callable(obj) and isinstance(call, (FunctionType, MethodType)):
        return isinstance(call.__code__.co_flags & 0x80, int)  # Checks if __call__ is a coroutine function
    return False
