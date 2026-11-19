from __future__ import annotations

import textwrap
from collections.abc import Sequence
from types import GenericAlias
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, overload
from warnings import warn

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

    class _ColorOpsMixin[OtherT]:
        def __add__(self, other: OtherT) -> Self: ...
        def __radd__(self, other: OtherT) -> Self: ...
        def __sub__(self, other: OtherT) -> Self: ...
        def __rsub__(self, other: OtherT) -> Self: ...
        def __mul__(self, other: OtherT) -> Self: ...
        def __rmul__(self, other: OtherT) -> Self: ...
        def __truediv__(self, other: OtherT) -> Self: ...
        def __rtruediv__(self, other: OtherT) -> Self: ...
        def __floordiv__(self, other: OtherT) -> Self: ...
        def __rfloordiv__(self, other: OtherT) -> Self: ...
        def __mod__(self, other: OtherT) -> Self: ...
        def __rmod__(self, other: OtherT) -> Self: ...
        def __pos__(self) -> Self: ...
        def __neg__(self) -> Self: ...
        def __abs__(self) -> Self: ...

    class _ColorIndexMixin[ItemT, SliceArrT]:
        @overload
        def __getitem__(self, key: int | np.integer[Any]) -> ItemT: ...
        @overload
        def __getitem__(self, key: slice | Sequence[int] | npt.NDArray[np.integer[Any] | np.bool_]) -> SliceArrT: ...
        def __getitem__(self, key: Any) -> Any: ...

else:

    class _ColorOpsMixin[OtherT]:
        pass

    class _ColorIndexMixin[ItemT, SliceArrT]:
        pass


class _GenericArrayMixin:
    """Ensure standard GenericAlias creation takes precedence over np.ndarray.__class_getitem__."""

    def __class_getitem__(cls, params: Any) -> Any:
        return GenericAlias(cls, params)


class _BaseRGBColor[
    DTypeT: np.floating[Any] | np.integer[Any],
    ItemT: float | int,
    C: Literal[3, 4],
](_GenericArrayMixin, np.ndarray[tuple[C], np.dtype[DTypeT]]):
    """Base immutable color class parameterized by NumPy dtype, Python scalar type, and channel count (3 or 4)."""

    _DTYPE: ClassVar[np.dtype[Any]]

    @classmethod
    def _ITEM_TYPE(cls) -> type[ItemT]:
        raise NotImplementedError

    @classmethod
    def channel_max_values(cls) -> tuple[ItemT, ...]:
        raise NotImplementedError

    @classmethod
    def num_channels(cls) -> int:
        return len(cls.channel_max_values())

    def __new__(cls, color: ColorLike) -> Self:
        return cls.as_color(color, copy=True)

    @classmethod
    def as_color(cls, color: ColorLike, copy: bool = False) -> Self:
        """Parse a color specification (RGB/RGBA tuple, ndarray, HEX string, or CSS color name)."""
        if isinstance(color, cls) and not copy:
            return color

        max_vals = np.asarray(cls.channel_max_values(), dtype=np.float64)
        n_channels = len(max_vals)
        is_float_target = np.issubdtype(cls._DTYPE, np.floating)

        if isinstance(color, str):
            s = color.strip().lower()
            if s.startswith("#"):
                hex_str = s[1:]
                if len(hex_str) in (3, 4):
                    raw = np.array([int(c * 2, 16) for c in hex_str], dtype=np.uint8)
                elif len(hex_str) in (6, 8):
                    raw = np.array([int(c, 16) for c in textwrap.wrap(hex_str, 2)], dtype=np.uint8)
                else:
                    raise ValueError(f"Invalid HEX color: {color!r}")
            else:
                import webcolors

                try:
                    raw = np.array(webcolors.name_to_rgb(s), dtype=np.uint8)
                except ValueError:
                    raise ValueError(f"Invalid color name: {color!r}") from None

            target_max = max_vals[: raw.size]
            if is_float_target:
                out = (raw.astype(np.float64) / 255.0) * target_max
            else:
                out = np.rint((raw.astype(np.float64) / 255.0) * target_max)

        elif isinstance(color, (Sequence, np.ndarray)):
            arr = np.asarray(color)
            if arr.ndim != 1 or arr.size not in (3, 4):
                raise ValueError(f"Invalid color shape: {arr.shape}. Must be 1D with 3 or 4 elements.")
            target_max = max_vals[: arr.size]
            if np.issubdtype(arr.dtype, np.floating):
                if is_float_target:
                    out = arr.astype(np.float64, copy=True)
                else:
                    out = arr.astype(np.float64) * target_max
            elif np.issubdtype(arr.dtype, np.integer):
                if is_float_target:
                    out = (arr.astype(np.float64) / 255.0) * target_max
                else:
                    out = arr.astype(np.float64, copy=True)
            else:
                raise TypeError(f"Invalid color dtype: {arr.dtype}. Must be integer or float.")
        else:
            raise TypeError(f"Invalid color specification: {color!r}")

        # Adapt channel count (3 <-> 4)
        if out.size == 3 and n_channels == 4:
            out = np.append(out, max_vals[3])
        elif out.size == 4 and n_channels == 3:
            warn(f"Alpha channel dropped when converting 4-channel color to {cls.__name__}.", UserWarning, stacklevel=2)
            out = out[:3].copy()

        return cls._wrap_as_color(out, copy=copy)

    @classmethod
    def _wrap_as_color(cls, arr: np.ndarray, *, copy: bool = False) -> Self:
        data = np.asarray(arr, dtype=np.float64)
        max_vals = np.asarray(cls.channel_max_values(), dtype=np.float64)
        clipped = np.clip(data, 0.0, max_vals)
        if not np.issubdtype(cls._DTYPE, np.floating):
            clipped = np.rint(clipped)
        out = clipped.astype(cls._DTYPE, copy=copy)
        out.flags.writeable = False
        return out.view(cls)

    def __array_finalize__(self, obj: Any) -> None:
        if obj is not None and np.shares_memory(self, obj):
            self.flags.writeable = False

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(f"{type(self).__name__} objects are immutable and do not support item assignment")

    def copy(self, order: Literal["C", "F", "A", "K"] | None = None) -> np.ndarray[tuple[C], np.dtype[DTypeT]]:
        """Return a mutable NumPy array copy of the color channels."""
        return self.view(np.ndarray).copy(order=order)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if "out" in kwargs:
            raise TypeError(f"{type(self).__name__} objects are immutable and do not support in-place operations")

        n_channels = self.num_channels()
        args = []
        for item in inputs:
            if isinstance(item, _BaseRGBColor):
                # Compute in float64 to prevent uint8 overflow/underflow before clipping
                args.append(item.view(np.ndarray).astype(np.float64))
            elif np.isscalar(item) and not isinstance(item, str):
                args.append(float(item))
            elif isinstance(item, (Sequence, np.ndarray)) and not isinstance(item, str):
                arr = np.asarray(item, dtype=np.float64)
                if arr.ndim == 1 and arr.size in (3, 4):
                    if arr.size == 3 and n_channels == 4:
                        arr = np.append(arr, 0.0)
                    elif arr.size == 4 and n_channels == 3:
                        arr = arr[:3]
                args.append(arr)
            else:
                args.append(item)

        result = getattr(ufunc, method)(*args, **kwargs)
        if result is NotImplemented:
            return NotImplemented

        def _wrap(res: Any) -> Any:
            if isinstance(res, np.ndarray) and res.shape == (n_channels,) and np.issubdtype(res.dtype, np.number):
                return type(self)._wrap_as_color(res)
            return res

        if isinstance(result, tuple):
            return tuple(_wrap(x) for x in result)
        return _wrap(result)

    if not TYPE_CHECKING:

        def __getitem__(self, key: Any) -> Any:
            result = super().__getitem__(key)
            if isinstance(result, _BaseRGBColor) and result.shape != (self._num_channels(),):
                return result.view(np.ndarray)
            return result

    def __eq__(self, other: object) -> bool:
        try:
            other_col = other if isinstance(other, type(self)) else type(self).as_color(other, copy=False)  # type: ignore[arg-type]
            return bool(np.allclose(self.view(np.ndarray), other_col.view(np.ndarray)))
        except (ValueError, TypeError):
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.tuple)

    def __str__(self) -> str:
        try:
            return self.to_name()
        except ValueError:
            return self.to_hex()

    def __repr__(self) -> str:
        return f'{type(self).__name__}("{self!s}")'

    @classmethod
    def as_item(cls, value: Any) -> ItemT:
        return cls._ITEM_TYPE()(value)

    @property
    def r(self) -> ItemT:
        return self.as_item(self.view(np.ndarray)[0])

    @property
    def g(self) -> ItemT:
        return self.as_item(self.view(np.ndarray)[1])

    @property
    def b(self) -> ItemT:
        return self.as_item(self.view(np.ndarray)[2])

    @property
    def tuple(self) -> tuple[ItemT, ...]:
        """Return the color channels as a Python tuple."""
        return tuple(self.as_item(c) for c in self.view(np.ndarray))

    def to_hex(self) -> str:
        """Convert the color to a HEX string (#rrggbb or #rrggbbaa)."""
        arr = self.view(np.ndarray).astype(np.float64)
        max_vals = np.asarray(self.channel_max_values(), dtype=np.float64)
        scaled = np.rint(np.clip(arr / max_vals, 0.0, 1.0) * 255).astype(np.uint8)
        return "#" + "".join(f"{int(c):02x}" for c in scaled)

    def to_name(self) -> str:
        """Convert the RGB channels to a CSS3 color name if an exact match exists."""
        import webcolors

        arr = self.view(np.ndarray)[:3].astype(np.float64)
        max_vals = np.asarray(self.channel_max_values()[:3], dtype=np.float64)
        scaled = np.rint(np.clip(arr / max_vals, 0.0, 1.0) * 255).astype(np.uint8)
        try:
            name = webcolors.rgb_to_name((int(scaled[0]), int(scaled[1]), int(scaled[2])))
        except ValueError:
            raise ValueError(f"Color {self.to_hex()} does not have a named representation.") from None
        if self.num_channels() == 4 and self.view(np.ndarray)[3] < max_vals[3]:
            return f"{name} (alpha={self.view(np.ndarray)[3] / max_vals[3]:.1%})"
        else:
            return name

    def as_rgb(self) -> RGB:
        """Convert to an 8-bit RGB color (dropping alpha if present)."""
        return RGB.as_color(self.tuple[:3], copy=True)

    def as_rgba(self, alpha: float | None = None) -> RGBA:
        """Convert to an 8-bit RGBA color, optionally overriding alpha (0-255 or 0.0-1.0)."""
        rgba = RGBA.as_color(self.tuple, copy=True)
        if alpha is None:
            return rgba
        a_val = round(alpha * 255) if isinstance(alpha, float) else int(alpha)
        arr = rgba.copy()
        arr[3] = np.clip(a_val, 0, 255)
        return RGBA._wrap_as_color(arr)

    def blend(self, other: ColorLike, alpha: float = 0.5) -> Self:
        """Linearly interpolate between self and other (alpha=0 -> self, alpha=1 -> other)."""
        other_col = type(self).as_color(other, copy=False)
        blended = (1.0 - alpha) * self.view(np.ndarray) + alpha * other_col.view(np.ndarray)
        return type(self)._wrap_as_color(blended)

    ###############################
    ##   Pydantic integration    ##
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(value: Any) -> Self:
            if isinstance(value, cls):
                return value
            try:
                return cls.as_color(value, copy=True)
            except TypeError as e:
                raise ValueError(str(e)) from e

        def serialize_hex(col: Self) -> str:
            hex_str = col.to_hex()
            if len(hex_str) == 9 and hex_str.endswith("ff"):
                return hex_str[:7]
            return hex_str

        n_channels = cls.num_channels()
        item_schema = core_schema.float_schema() if np.issubdtype(cls._DTYPE, np.floating) else core_schema.int_schema()
        tuple_schema = core_schema.tuple_schema([item_schema] * n_channels)
        json_input_schemas: list[CoreSchema] = [tuple_schema, core_schema.str_schema()]
        if n_channels == 4:
            json_input_schemas.insert(1, core_schema.tuple_schema([item_schema] * 3))

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                validate,
                core_schema.union_schema(json_input_schemas),
            ),
            python_schema=core_schema.no_info_plain_validator_function(validate),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_hex,
                return_schema=core_schema.str_schema(),
            ),
        )


class RGB(
    _ColorOpsMixin["RGBColorLike | int | float"],
    _ColorIndexMixin[int, np.ndarray[Any, np.dtype[np.uint8]]],
    _BaseRGBColor[np.uint8, int, Literal[3]],
):
    """Immutable 8-bit RGB color (r, g, b) with values in [0, 255]."""

    _DTYPE = np.dtype(np.uint8)

    @classmethod
    def _ITEM_TYPE(cls) -> type[int]:
        return int

    @classmethod
    def channel_max_values(cls) -> tuple[int, int, int]:
        return (255, 255, 255)

    @property
    def tuple(self) -> tuple[int, int, int]:
        arr = self.view(np.ndarray)
        return (int(arr[0]), int(arr[1]), int(arr[2]))


class RGBA(
    _ColorOpsMixin["RGBAColorLike | int | float"],
    _ColorIndexMixin[int, np.ndarray[Any, np.dtype[np.uint8]]],
    _BaseRGBColor[np.uint8, int, Literal[4]],
):
    """Immutable 8-bit RGBA color (r, g, b, a) with values in [0, 255]."""

    _DTYPE = np.dtype(np.uint8)

    @classmethod
    def _ITEM_TYPE(cls) -> type[int]:
        return int

    @classmethod
    def channel_max_values(cls) -> tuple[int, int, int, int]:
        return (255, 255, 255, 255)

    @property
    def a(self) -> int:
        return int(self.view(np.ndarray)[3])

    @property
    def tuple(self) -> tuple[int, int, int, int]:
        arr = self.view(np.ndarray)
        return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))


type RGBColorLikeStrict = (
    RGB | tuple[int, int, int] | tuple[float, float, float] | np.ndarray[tuple[Literal[3]], np.dtype[np.number]]
)
type RGBAColorLikeStrict = (
    RGBA
    | tuple[int, int, int, int]
    | tuple[float, float, float, float]
    | np.ndarray[tuple[Literal[4]], np.dtype[np.number]]
)

type RGBColorLike = RGBColorLikeStrict | RGBAColorLikeStrict | str
type RGBAColorLike = RGBAColorLikeStrict | RGBColorLikeStrict | str
type ColorLike = RGBAColorLike
